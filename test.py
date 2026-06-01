import os
import math
import time
import torch
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
from utils.config import load_config
from models.registry import get_model_cls


def run_forward(model, lr_clip, precision):
    """
    Run one model forward pass on a clip.

    Parameters:
        model: Loaded generator model.
        lr_clip: Input clip on the active device.
        precision: Inference precision mode.

    Returns:
        Super-resolved clip shaped like (T, 3, H, W) in float32.
    """
    with torch.inference_mode():
        if precision == "fp32":
            pred = model(lr_clip)
        elif precision == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = model(lr_clip)
        elif precision == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(lr_clip)
        else:
            raise ValueError(f"Unsupported precision mode: {precision}")

    # Some models return (sr, cleaned_lq); inference keeps the SR output.
    sr_clip = pred[0] if isinstance(pred, (tuple, list)) else pred
    return sr_clip.clamp(0, 1).float().squeeze(0)


def tiled_inference(model, clip, tile_size, pad_size, scale, precision):
    """
    Run tiled inference over time and space.

    Parameters:
        model: Loaded generator model.
        clip: Input low-resolution clip on the active device.
        tile_size: Tuple of temporal and spatial tile sizes.
        pad_size: Tuple of temporal and spatial overlap sizes.
        scale: Spatial upsampling factor.
        precision: Inference precision mode.

    Returns:
        Full super-resolved clip on CPU.
    """
    tt, th, tw = tile_size
    pt, ph, pw = pad_size
    _, T, _, H, W = clip.shape

    tt = tt or T
    th = th or H
    tw = tw or W

    stride_t, stride_h, stride_w = tt, th, tw
    out_H, out_W = H * scale, W * scale

    out = torch.zeros((T, 3, out_H, out_W), dtype=torch.float32, device="cpu")
    cnt = torch.zeros_like(out)

    def _starts(full, tile):
        """
        Build tile start indices for one dimension.

        Parameters:
            full: Full length of the dimension.
            tile: Tile length for the dimension.

        Returns:
            List of starting indices.
        """
        if full <= tile:
            return [0]
        n = math.ceil(full / tile)
        return [min(i * tile, full - tile) for i in range(n)]

    t_starts = _starts(T, stride_t)
    h_starts = _starts(H, stride_h)
    w_starts = _starts(W, stride_w)

    for t0 in t_starts:
        for h0 in h_starts:
            for w0 in w_starts:
                t1 = min(t0 + tt, T)
                h1 = min(h0 + th, H)
                w1 = min(w0 + tw, W)

                t0_in = max(0, t0 - pt)
                h0_in = max(0, h0 - ph)
                w0_in = max(0, w0 - pw)
                t1_in = min(T, t1 + pt)
                h1_in = min(H, h1 + ph)
                w1_in = min(W, w1 + pw)

                lr_patch = clip[:, t0_in:t1_in, :, h0_in:h1_in, w0_in:w1_in]
                lr_patch = lr_patch.contiguous().clone()

                sr_patch = run_forward(model, lr_patch, precision).cpu()

                shave_t0 = t0 - t0_in
                shave_t1 = t1_in - t1
                shave_h0 = (h0 - h0_in) * scale
                shave_h1 = (h1_in - h1) * scale
                shave_w0 = (w0 - w0_in) * scale
                shave_w1 = (w1_in - w1) * scale

                sr_patch = sr_patch[
                    shave_t0: sr_patch.shape[0] - shave_t1 or None,
                    :,
                    shave_h0: sr_patch.shape[-2] - shave_h1 or None,
                    shave_w0: sr_patch.shape[-1] - shave_w1 or None,
                ]

                t_dst0 = t0
                h_dst0 = h0 * scale
                w_dst0 = w0 * scale

                t_dst1 = t_dst0 + sr_patch.shape[0]
                h_dst1 = h_dst0 + sr_patch.shape[-2]
                w_dst1 = w_dst0 + sr_patch.shape[-1]

                out[t_dst0:t_dst1, :, h_dst0:h_dst1, w_dst0:w_dst1].add_(sr_patch)
                cnt[t_dst0:t_dst1, :, h_dst0:h_dst1, w_dst0:w_dst1].add_(1)

    cnt.clamp_min_(1e-9)
    out.div_(cnt)
    return out


def validate_runtime_args(device, precision, compile_mode):
    """
    Validate precision and compile settings against the runtime.

    Parameters:
        device: Torch device used for inference.
        precision: Inference precision mode.
        compile_mode: torch.compile mode.

    Returns:
        Nothing. Raises an error if the settings are invalid.
    """
    if precision != "fp32" and device.type != "cuda":
        raise ValueError(
            f"--precision {precision} requires CUDA, but got device='{device.type}'."
        )

    if compile_mode != "none" and not hasattr(torch, "compile"):
        raise RuntimeError(
            "--compile_mode requires torch.compile, which is unavailable in this "
            f"runtime (PyTorch {torch.__version__}). Install a PyTorch 2.x build "
            "per the project setup instructions."
        )


def strip_leading_module_prefixes(key):
    """
    Strip leading "module." prefixes from a state dict key.

    It handles:
    - DDP(model): keys like "module.conv.weight"
    - DDP(CheckpointWrapper(model)): keys like "module.module.conv.weight"
    - CheckpointWrapper(model): keys like "module.conv.weight" because the wrapper
        stores the wrapped model in self.module

    Parameters:
        key: State dict key.

    Returns:
        Key with any number of leading "module." prefixes removed.
    """
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def maybe_compile_model(model, compile_mode):
    """
    Optionally wrap the model with torch.compile.

    Parameters:
        model: Loaded generator model.
        compile_mode: torch.compile mode.

    Returns:
        Compiled model if requested, otherwise the original model.
    """
    if compile_mode == "none":
        return model

    try:
        model = torch.compile(model, mode=compile_mode)
    except Exception as e:
        raise RuntimeError(
            f"Failed to compile model with --compile_mode {compile_mode!r}: {e}"
        ) from e

    return model


def super_resolve_and_save_sequences(
    model_path,
    config,
    tile_size=(0, 0, 0),
    pad_size=(0, 0, 0),
    precision="fp32",
    compile_mode="none",
):
    """
    Run video super-resolution inference sequence by sequence.

    Parameters:
        model_path: Path to the model checkpoint.
        config: Loaded YAML config dictionary.
        tile_size: Tuple of temporal and spatial tile sizes.
        pad_size: Tuple of temporal and spatial overlap sizes.
        precision: Inference precision mode.
        compile_mode: torch.compile mode.

    Returns:
        Nothing. Saves output frames to the configured output directory.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_runtime_args(DEVICE, precision, compile_mode)

    ckpt = torch.load(model_path, map_location=DEVICE)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    stripped_sd = {strip_leading_module_prefixes(k): v for k, v in raw_sd.items()}

    model_name = config["model"]["name"]
    model_cls = get_model_cls(model_name)
    model_kwargs = config["model"].copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("flow_param_keywords", None)

    model = model_cls(**model_kwargs).to(DEVICE)
    model.load_state_dict(stripped_sd, strict=False)
    model.eval()
    model = maybe_compile_model(model, compile_mode)

    if compile_mode != "none":
        print(
            f"[INFO] torch.compile enabled with mode='{compile_mode}'. "
            "The first sequence timing includes compilation overhead."
        )

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    input_root = config["dataset"]["test"]["lr_dir"]
    output_root = config["dataset"]["test"]["hr_dir"]

    seq_iter = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    seq_iter = tqdm(seq_iter, desc="Sequences", unit="seq")
    scale = model_kwargs.get("scale_factor", 4)

    for seq_name in seq_iter:
        in_dir = os.path.join(input_root, seq_name)
        out_dir = os.path.join(output_root, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        frame_files = sorted(
            f for f in os.listdir(in_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not frame_files:
            tqdm.write(f"[WARNING] {seq_name}: no frames found, skipped")
            continue

        valid_frames = []
        for frame_name in frame_files:
            try:
                img = Image.open(os.path.join(in_dir, frame_name)).convert("RGB")
                valid_frames.append(to_tensor(img))
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to read {frame_name}: {e}")

        if not valid_frames:
            tqdm.write(f"[ERROR] All frames in {seq_name} are unreadable, skipping")
            continue

        lr_clip = torch.stack(valid_frames, dim=0).unsqueeze(0).to(DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        if any(tile_size):
            sr_clip = tiled_inference(
                model,
                lr_clip,
                tile_size,
                pad_size,
                scale,
                precision,
            )
        else:
            sr_clip = run_forward(model, lr_clip, precision).cpu()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        avg_time = elapsed / len(frame_files) if frame_files else float("nan")

        frame_iter = tqdm(
            enumerate(sr_clip),
            total=len(sr_clip),
            desc=f"Saving {seq_name}",
            leave=False,
            unit="frame",
        )

        for i, frame in frame_iter:
            to_pil(frame.cpu()).save(os.path.join(out_dir, f"{i:08d}.png"))

        del lr_clip, sr_clip
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        tqdm.write(
            f"[DONE] {seq_name}: {len(frame_files)} frames -> {out_dir} | "
            f"Avg inference {avg_time:.4f} sec/frame"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run video super-resolution inference on the test set using a trained model and a config file.\n\n"
            "The config YAML must specify the model architecture and the input/output directories for testing.\n"
            "If --tile_t, --tile_h, and --tile_w are all 0, full-clip inference is used. "
            "Otherwise, tiled inference is used.\n\n"
            "Example usage:\n"
            "  python test.py configs/PAWS_BasicVSRPP_PSNR_REDSx4.yaml outputs/PAWS_BasicVSRPP_PSNR_REDSx4_G.pth\n\n"
            "Required config keys (YAML example):\n"
            "  model:\n"
            "    name: BasicVSRPlusPlus\n"
            "    ...\n"
            "  dataset:\n"
            "    test:\n"
            "      lr_dir: path/to/low_res_frames/\n"
            "      hr_dir: path/to/save_high_res_frames/"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("config", type=str, help="Path to the YAML config file used for training (should include model definition and test dataset directories)")
    parser.add_argument("model_path", type=str, help="Path to the model checkpoint")
    parser.add_argument("--tile_t", type=int, default=0, help="Temporal tile length (0 = full)")
    parser.add_argument("--tile_h", type=int, default=0, help="Spatial tile height (0 = full)")
    parser.add_argument("--tile_w", type=int, default=0, help="Spatial tile width (0 = full)")
    parser.add_argument("--pad_t", type=int, default=0, help="Temporal overlap / padding")
    parser.add_argument("--pad_h", type=int, default=20, help="Spatial overlap / padding height (default: 20)")
    parser.add_argument("--pad_w", type=int, default=20, help="Spatial overlap / padding width (default: 20)")
    parser.add_argument(
        "--precision",
        type=str,
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
        help="Inference precision mode; bf16/fp16 require CUDA (default: fp32).",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
        help="torch.compile mode to use (default: none)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    super_resolve_and_save_sequences(
        args.model_path,
        config,
        (args.tile_t, args.tile_h, args.tile_w),
        (args.pad_t, args.pad_h, args.pad_w),
        args.precision,
        args.compile_mode,
    )
