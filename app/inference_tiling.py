"""
Handle tile sizing, tiled inference, and retry logic.
"""

import math

import torch
from tqdm import tqdm

from constants import MIN_SPATIAL_TILE
from constants import MIN_TEMPORAL_TILE
from inference_runtime import clear_cuda_runtime_cache
from inference_runtime import compact_exception_message
from inference_runtime import disable_compiled_model
from inference_runtime import get_active_model
from inference_runtime import is_cuda_out_of_memory
from inference_runtime import is_torch_compile_failure
from inference_runtime import log_runtime_message
from inference_runtime import maybe_log_first_compiled_forward
from inference_runtime import notify_runtime_status
from inference_runtime import run_forward


def build_tile_starts(full, tile):
    """
    Build tile start indices for one dimension.

    Args:
        full (int): Full dimension length.
        tile (int): Tile length.

    Returns:
        list: Tile start indices.
    """
    if full <= tile:
        return [0]
    count = math.ceil(full / tile)
    return [min(index * tile, full - tile) for index in range(count)]


def resolve_tile_length(requested, full, minimum, label):
    """
    Resolve one runtime tile length (temporal or spatial).

    Args:
        requested (int): User-requested tile size.
        full (int): Full source dimension.
        minimum (int): Minimum allowed tile size.
        label (str): Axis label used in error messages.

    Returns:
        int: Positive tile size.

    Raises:
        ValueError: If the requested tile is below the minimum.
    """
    requested = int(requested or 0)
    full = max(1, int(full or 1))

    if requested < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")

    return max(1, min(requested, full))


def resolve_runtime_tiling(model_settings, height, width, total_frames):
    """
    Resolve raw UI tile settings into runtime values.

    Args:
        model_settings (dict): Inference settings dictionary.
        height (int): Source frame height.
        width (int): Source frame width.
        total_frames (int): Estimated total frame count.

    Returns:
        dict: Settings dictionary with resolved tile values.
    """

    def _read_int(key, default):
        try:
            return int(model_settings.get(key, default) or default)
        except Exception:
            return int(default)

    requested_t = _read_int("tile_t", 0)
    requested_h = _read_int("tile_h", 0)
    requested_w = _read_int("tile_w", 0)

    resolved = dict(model_settings)
    resolved["resolved_tile_t"] = int(
        resolve_tile_length(
            requested_t, total_frames, MIN_TEMPORAL_TILE, "Temporal tile length"
        )
    )
    resolved["resolved_tile_h"] = int(
        resolve_tile_length(
            requested_h, height, MIN_SPATIAL_TILE, "Spatial tile height"
        )
    )
    resolved["resolved_tile_w"] = int(
        resolve_tile_length(requested_w, width, MIN_SPATIAL_TILE, "Spatial tile width")
    )
    resolved["tiling_message"] = ""
    return resolved


def tiled_inference(model, clip, tile_size, pad_size, scale, precision, device):
    """
    Run tiled inference over time and space for one clip.

    Args:
        model (torch.nn.Module): Loaded generator model.
        clip (torch.Tensor): Input clip shaped like ``B,T,C,H,W``.
        tile_size (tuple): Tuple of temporal and spatial tile sizes.
        pad_size (tuple): Tuple of temporal and spatial padding sizes.
        scale (int): Spatial scale factor.
        precision (str): Inference precision mode.
        device (torch.device): Active torch device.

    Returns:
        torch.Tensor: Super-resolved clip on CPU.
    """
    tile_t, tile_h, tile_w = tile_size
    pad_t, pad_h, pad_w = pad_size
    _, frame_count, _, frame_height, frame_width = clip.shape

    tile_t = max(1, min(int(tile_t), int(frame_count)))
    tile_h = max(1, min(int(tile_h), int(frame_height)))
    tile_w = max(1, min(int(tile_w), int(frame_width)))

    output_height = frame_height * scale
    output_width = frame_width * scale
    output_clip = torch.zeros(
        (frame_count, 3, output_height, output_width), dtype=torch.float32, device="cpu"
    )
    overlap_count = torch.zeros_like(output_clip)

    temporal_starts = build_tile_starts(frame_count, tile_t)
    row_starts = build_tile_starts(frame_height, tile_h)
    col_starts = build_tile_starts(frame_width, tile_w)
    tile_count = len(temporal_starts) * len(row_starts) * len(col_starts)
    tile_progress = tqdm(
        total=tile_count,
        desc=f"PAWS window tiles T={tile_t} H={tile_h} W={tile_w}",
        unit="tile",
        leave=False,
        dynamic_ncols=True,
    )

    try:
        for t_start in temporal_starts:
            for h_start in row_starts:
                for w_start in col_starts:
                    t_end = min(t_start + tile_t, frame_count)
                    h_end = min(h_start + tile_h, frame_height)
                    w_end = min(w_start + tile_w, frame_width)

                    # Expand the tile by the requested padding so the model sees
                    # overlap context around each tile boundary.
                    t_input_start = max(0, t_start - pad_t)
                    h_input_start = max(0, h_start - pad_h)
                    w_input_start = max(0, w_start - pad_w)
                    t_input_end = min(frame_count, t_end + pad_t)
                    h_input_end = min(frame_height, h_end + pad_h)
                    w_input_end = min(frame_width, w_end + pad_w)

                    lr_patch = clip[
                        :,
                        t_input_start:t_input_end,
                        :,
                        h_input_start:h_input_end,
                        w_input_start:w_input_end,
                    ]
                    lr_patch = lr_patch.contiguous().to(device)
                    try:
                        sr_patch = run_forward(model, lr_patch, precision).cpu()
                    finally:
                        del lr_patch

                    # Crop the padded margins back off before stitching the patch
                    # into the final output tensor.
                    shave_t_start = t_start - t_input_start
                    shave_t_end = t_input_end - t_end
                    shave_h_start = (h_start - h_input_start) * scale
                    shave_h_end = (h_input_end - h_end) * scale
                    shave_w_start = (w_start - w_input_start) * scale
                    shave_w_end = (w_input_end - w_end) * scale

                    sr_patch = sr_patch[
                        shave_t_start : sr_patch.shape[0] - shave_t_end or None,
                        :,
                        shave_h_start : sr_patch.shape[-2] - shave_h_end or None,
                        shave_w_start : sr_patch.shape[-1] - shave_w_end or None,
                    ]

                    t_output_start = t_start
                    h_output_start = h_start * scale
                    w_output_start = w_start * scale
                    t_output_end = t_output_start + sr_patch.shape[0]
                    h_output_end = h_output_start + sr_patch.shape[-2]
                    w_output_end = w_output_start + sr_patch.shape[-1]

                    # Overlapping padded tiles contribute to the same output
                    # pixels, so accumulate both values and write counts.
                    output_clip[
                        t_output_start:t_output_end,
                        :,
                        h_output_start:h_output_end,
                        w_output_start:w_output_end,
                    ].add_(sr_patch)
                    overlap_count[
                        t_output_start:t_output_end,
                        :,
                        h_output_start:h_output_end,
                        w_output_start:w_output_end,
                    ].add_(1)
                    tile_progress.update(1)
    finally:
        tile_progress.close()

    # Average overlapping writes from padded tiles back into one output clip.
    overlap_count.clamp_min_(1e-9)
    output_clip.div_(overlap_count)
    return output_clip


def resolve_temporal_chunk_size(model_settings):
    """
    Resolve the streaming chunk size from settings.

    Args:
        model_settings (dict): Inference settings dictionary.

    Returns:
        int: Number of core frames per chunk.

    Raises:
        ValueError: If the effective temporal tile is invalid.
    """
    if "resolved_tile_t" in model_settings:
        requested = int(model_settings.get("resolved_tile_t", 0) or 0)
        if requested <= 0:
            raise ValueError("Resolved temporal tile length is invalid.")
        return requested

    requested = int(model_settings.get("tile_t", 0) or 0)
    if requested < MIN_TEMPORAL_TILE:
        raise ValueError(f"Temporal tile length must be at least {MIN_TEMPORAL_TILE}.")
    return requested


def make_frame_record(frame):
    """
    Convert a decoded frame into a tensor record.

    Args:
        frame (av.VideoFrame): Input PyAV video frame.

    Returns:
        dict: Frame tensor and timing metadata.
    """
    rgb = frame.to_ndarray(format="rgb24")
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
    return {
        "tensor": tensor,
        "pts": frame.pts,
        "time_base": frame.time_base,
    }


def resolve_window_tiling(model_settings, frames, height, width):
    """
    Build tile and padding sizes for one inference window.

    Args:
        model_settings (dict): Inference settings dictionary.
        frames (int): Number of frames in the window.
        height (int): Frame height.
        width (int): Frame width.

    Returns:
        tuple: Tile-size tuple and padding-size tuple.
    """
    resolved_tile_t = int(model_settings.get("resolved_tile_t", 0) or 0)
    resolved_tile_h = int(model_settings.get("resolved_tile_h", 0) or 0)
    resolved_tile_w = int(model_settings.get("resolved_tile_w", 0) or 0)

    if resolved_tile_t <= 0:
        resolved_tile_t = resolve_tile_length(
            model_settings.get("tile_t", 0),
            frames,
            MIN_TEMPORAL_TILE,
            "Temporal tile length",
        )
    if resolved_tile_h <= 0:
        resolved_tile_h = resolve_tile_length(
            model_settings.get("tile_h", 0),
            height,
            MIN_SPATIAL_TILE,
            "Spatial tile height",
        )
    if resolved_tile_w <= 0:
        resolved_tile_w = resolve_tile_length(
            model_settings.get("tile_w", 0),
            width,
            MIN_SPATIAL_TILE,
            "Spatial tile width",
        )

    tile_size = (resolved_tile_t, resolved_tile_h, resolved_tile_w)
    pad_size = (
        int(model_settings.get("pad_t", 0) or 0),
        int(model_settings.get("pad_h", 0) or 0),
        int(model_settings.get("pad_w", 0) or 0),
    )
    return tile_size, pad_size


def infer_window(model, records, model_settings, scale, device):
    """
    Run inference for one bounded temporal window.

    Args:
        model (torch.nn.Module): Loaded generator model.
        records (list): Frame records for the window.
        model_settings (dict): Inference settings dictionary.
        scale (int): Model scale factor.
        device (torch.device): Active torch device.

    Returns:
        torch.Tensor: Super-resolved clip on CPU.
    """
    clip = torch.stack([record["tensor"] for record in records], dim=0).unsqueeze(0)
    precision = model_settings.get("precision", "fp32")
    frames = int(clip.shape[1])
    height = int(clip.shape[-2])
    width = int(clip.shape[-1])
    tile_size, pad_size = resolve_window_tiling(model_settings, frames, height, width)

    try:
        return tiled_inference(
            model, clip, tile_size, pad_size, int(scale), precision, device
        )
    finally:
        del clip
        if device.type == "cuda":
            torch.cuda.empty_cache()


def retry_without_compile(model_runtime, progress_cb, total_frames, processed, reason):
    """
    Disable compile mode, log the reason, and tell the UI about the retry.

    Args:
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        processed (int): Number of written frames so far.
        reason (str): Retry message.

    Returns:
        bool: True when compile mode was disabled.
    """
    if not disable_compiled_model(model_runtime):
        return False
    log_runtime_message("compile", reason)
    notify_runtime_status(progress_cb, total_frames, reason, processed)
    return True


def infer_window_with_retry(
    model_runtime,
    records,
    model_settings,
    scale,
    device,
    progress_cb,
    total_frames,
    processed,
):
    """
    Run one window with compile fallback and OOM retries.

    Args:
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.
        records (list): Frame records for the window.
        model_settings (dict): Runtime inference settings.
        scale (int): Model scale factor.
        device (torch.device): Active torch device.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        processed (int): Number of written frames so far.

    Returns:
        torch.Tensor: Super-resolved clip on CPU.

    Raises:
        RuntimeError: If eager retry still fails because of CUDA memory limits.
    """
    while True:
        try:
            maybe_log_first_compiled_forward(
                model_runtime, progress_cb, total_frames, processed
            )
            return infer_window(
                get_active_model(model_runtime), records, model_settings, scale, device
            )
        except Exception as exc:
            if device.type == "cuda" and is_cuda_out_of_memory(exc):
                clear_cuda_runtime_cache(device)

                # Under compile mode, try the cheaper eager retry path before
                # surfacing the user-facing tile and precision guidance error.
                if retry_without_compile(
                    model_runtime,
                    progress_cb,
                    total_frames,
                    processed,
                    "CUDA OOM under torch.compile; disabling compile mode and retrying eager inference.",
                ):
                    continue

                raise RuntimeError(
                    "CUDA out of memory with the current tile settings. "
                    "Try smaller temporal or spatial tiles, lower padding, fp16 or bf16 precision, "
                    "or compile mode 'none'."
                ) from exc

            if is_torch_compile_failure(exc):
                message = (
                    "torch.compile failed during inference; disabling compile mode "
                    f"and retrying eager inference. Reason: {compact_exception_message(exc)}"
                )
                # TorchInductor failures are retried once in eager mode so a
                # compile-only issue does not abort the whole run.
                if retry_without_compile(
                    model_runtime,
                    progress_cb,
                    total_frames,
                    processed,
                    message,
                ):
                    continue

            raise
