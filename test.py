import os
import torch
import time
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
from utils.config import load_config
from models.factory import MODEL_REGISTRY


def super_resolve_and_save_sequences(model_path: str, config: dict):
    """
    Run video super-resolution inference on a dataset directory and saving the results sequence-wise.

    Loads a trained generator model and applies it to all video sequences in the test set
    specified by the config. Each sequence directory is processed independently: all frames
    are stacked, passed through the model, and super-resolved outputs are saved as images
    in the output directory. For each sequence, the function also prints the average inference
    time per frame (including CUDA synchronization if applicable).

    Args:
        model_path (str): Path to the saved model checkpoint (expects 'model_state_dict' or bare state dict).
        config (dict): Configuration dictionary, must include keys for model instantiation
            and test dataset paths:
                - config["model"]: Dict specifying model architecture and constructor args.
                - config["dataset"]["test"]["lr_dir"]: Directory with input LR frames (one subdir per sequence).
                - config["dataset"]["test"]["hr_dir"]: Output directory to save SR frames.

    Raises:
        ValueError: If the specified model architecture name is not registered.

    Notes:
        - Handles models saved with or without DistributedDataParallel/CheckpointWrapper prefixes.
        - Output SR frames are saved in 8-digit zero-padded PNG filenames (00000000.png, ...).
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint and remove any "module." prefixes (handles DDP/wrappers)
    ckpt = torch.load(model_path, map_location=DEVICE)
    raw_sd = ckpt.get("model_state_dict", ckpt)
    stripped_sd = {k.replace("module.", ""): v for k, v in raw_sd.items()}

    # Build model instance from config
    model_name = config["model"]["name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown generator model name: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]
    model_kwargs = config["model"].copy()
    model_kwargs.pop("name", None)
    model_kwargs.pop("flow_param_keywords", None)

    model = model_cls(**model_kwargs).to(DEVICE)
    model.load_state_dict(stripped_sd, strict=False)
    model.eval()

    # Basic image transforms (torch <-> PIL)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()


    input_root = config["dataset"]["test"]["lr_dir"]
    output_root = config["dataset"]["test"]["hr_dir"]

    seq_iter = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    seq_iter = tqdm(seq_iter, desc="Sequences", unit="seq")

    # Loop over each test sequence in LR directory
    for seq_name in seq_iter:
        in_dir  = os.path.join(input_root,  seq_name)
        out_dir = os.path.join(output_root, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        frame_files = sorted(f for f in os.listdir(in_dir)
                             if f.lower().endswith((".png", ".jpg", ".jpeg")))
        if not frame_files:
            tqdm.write(f"[WARNING] {seq_name}: no frames found, skipped")
            continue

        # Stack all frames for the sequence: shape (1, T, 3, H, W)
        valid_frames = []
        for f in frame_files:
            try:
                img = Image.open(os.path.join(in_dir, f)).convert("RGB")
                valid_frames.append(to_tensor(img))
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to read {f}: {e}")
        if not valid_frames:
            tqdm.write(f"[ERROR] All frames in {seq_name} are unreadable, skipping")
            continue
        lr_clip = torch.stack(valid_frames, dim=0).unsqueeze(0).to(DEVICE)

        # Run model inference
        if DEVICE.type == "cuda":
            # Synchronize CUDA before/after inference to ensure timing measures only the model forward pass (no-op on CPU)
            torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            sr_clip = model(lr_clip).clamp_(0, 1).squeeze(0)  # (T, 3, H_s, W_s)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time
        avg_time = elapsed / len(frame_files) if frame_files else float('nan')

        frame_iter = tqdm(enumerate(sr_clip),
                          total=len(sr_clip),
                          desc=f"Saving {seq_name}",
                          leave=False,
                          unit="frame")

        # Save super-resolved frames to output directory, sequentially
        for i, fr in frame_iter:
            to_pil(fr.cpu()).save(os.path.join(out_dir, f"{i:08d}.png"))

        tqdm.write(f"[DONE] {seq_name}: {len(frame_files)} frames -> {out_dir} | "
                   f"Avg inference {avg_time:.4f} sec/frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run video super-resolution inference on the test set using a trained model and a config file.\n\n"
            "The config YAML must specify the model architecture and the input/output directories for testing.\n"
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
    args = parser.parse_args()

    config = load_config(args.config)

    super_resolve_and_save_sequences(model_path=args.model_path, config=config)
