"""
Compute Ewarp over generated video sequence folders.

This script scans a directory of sequence subfolders, computes flow warping
error over adjacent generated frames, and builds a report with per-sequence
scores plus overall averages.

Purpose:
- Score temporal self-consistency without requiring ground-truth frames.
- Print E*warp by default, where E*warp = Ewarp * 1000.
- Print a human-readable summary and optionally write JSON output.

Examples:
  python -m tools.compute_ewarp data/REDS/hr/test
  python -m tools.compute_ewarp data/REDS/hr/test \
    --json-out outputs/ewarp_report.json
  python -m tools.compute_ewarp data/REDS/hr/test --flow-model raft \
    --flow-ckpt sintel
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import ptlflow
import torch
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_flow_model(flow_model, flow_ckpt, device):
    """
    Create a PTLFlow optical-flow model.

    Args:
        flow_model (str): PTLFlow model name.
        flow_ckpt (str): PTLFlow checkpoint name or path.
        device (torch.device): Device used to host the model.

    Returns:
        torch.nn.Module: Initialized optical-flow model in eval mode.

    Raises:
        ValueError: If the requested model is unsupported.
    """
    available_flows = ptlflow.get_model_names()
    if flow_model not in available_flows:
        raise ValueError(
            "Optical flow model '{}' is not supported.\n"
            "Available options: {}".format(
                flow_model,
                available_flows,
            )
        )

    model = ptlflow.get_model(flow_model, ckpt_path=flow_ckpt)
    model = model.to(device)
    model.eval()
    return model


def load_frame(path, device):
    """
    Load one frame as an RGB float tensor.

    Args:
        path (str): Image file path.
        device (torch.device): Destination device.

    Returns:
        torch.Tensor or None: Tensor shaped (1, 3, H, W) in [0, 1], or None if
        the image cannot be read.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def make_pixel_grid(batch, height, width, device, dtype):
    """
    Build a pixel-coordinate grid shaped (B, H, W, 2).

    Args:
        batch (int): Batch dimension for the expanded grid.
        height (int): Grid height.
        width (int): Grid width.
        device (torch.device): Device used for the grid tensor.
        dtype (torch.dtype): Floating-point dtype used for coordinates.

    Returns:
        torch.Tensor: Pixel-coordinate grid shaped (B, H, W, 2), where the
        last dimension stores coordinates as (x, y).
    """
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, height, device=device, dtype=dtype),
        torch.arange(0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)


def flow_to_grid(flow):
    """
    Convert pixel-space flow shaped (B, 2, H, W) to a normalized sampling grid.

    Args:
        flow (torch.Tensor): Pixel-space flow shaped (B, 2, H, W), with
        channels ordered as (x, y) offsets.

    Returns:
        tuple: `(normalized_grid, pixel_grid_with_flow)`. The normalized grid
        is shaped (B, H, W, 2) in the [-1, 1] range expected by
        `grid_sample`; the pixel grid is shaped (B, H, W, 2) and stores
        `p + flow(p)` in pixel coordinates.
    """
    batch, _, height, width = flow.shape
    flow_hw = flow.permute(0, 2, 3, 1)
    coords = (
        make_pixel_grid(batch, height, width, flow.device, flow.dtype)
        + flow_hw
    )

    grid_x = 2.0 * coords[:, :, :, 0] / max(width - 1, 1) - 1.0
    grid_y = 2.0 * coords[:, :, :, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid, coords


def warp_tensor(x, flow):
    """
    Warp a tensor using pixel-space optical flow.

    Args:
        x (torch.Tensor): Tensor shaped (B, C, H, W).
        flow (torch.Tensor): Flow shaped (B, 2, H, W).

    Returns:
        torch.Tensor: Warped tensor shaped like x.
    """
    grid, _ = flow_to_grid(flow)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


def valid_coordinate_mask(flow):
    """
    Return a mask for pixels whose warped coordinates stay inside the frame.

    Args:
        flow (torch.Tensor): Pixel-space flow shaped (B, 2, H, W).

    Returns:
        torch.Tensor: Boolean mask shaped (B, H, W). A value is True when
        warped coordinate p' = p + flow(p) satisfies
        `0 <= x' <= W - 1` and `0 <= y' <= H - 1`.
    """
    _, coords = flow_to_grid(flow)
    _, _, height, width = flow.shape
    x = coords[:, :, :, 0]
    y = coords[:, :, :, 1]
    return (x >= 0.0) & (x <= width - 1) & (y >= 0.0) & (y <= height - 1)


def motion_boundary_mask(flow):
    """
    Detect invalid motion-boundary pixels.

    Ruder, Dosovitskiy, and Brox, "Artistic Style Transfer for Videos",
    GCPR 2016.

    For optical flow w = (u, v) in the current coordinate frame:
        grad_sq = ||grad u||^2 + ||grad v||^2
        flow_mag_sq = ||w||^2

    This returns True for pixels satisfying the invalid-boundary test:
        grad_sq > 0.01 * flow_mag_sq + 0.002
    """
    batch, _, height, width = flow.shape
    grad_sq = torch.zeros(
        (batch, height, width),
        device=flow.device,
        dtype=flow.dtype,
    )

    if width > 1:
        diff_x = flow[:, :, :, 1:] - flow[:, :, :, :-1]
        grad_sq[:, :, :-1] += diff_x.pow(2).sum(dim=1)

    if height > 1:
        diff_y = flow[:, :, 1:, :] - flow[:, :, :-1, :]
        grad_sq[:, :-1, :] += diff_y.pow(2).sum(dim=1)

    flow_mag_sq = flow.pow(2).sum(dim=1)
    return grad_sq > (0.01 * flow_mag_sq + 0.002)


def non_occlusion_mask(flow_forward, flow_backward):
    """
    Compute a binary non-occlusion mask using forward-backward consistency.

    Ruder, Dosovitskiy, and Brox, "Artistic Style Transfer for Videos",
    GCPR 2016.

    Let:
        f = flow_forward, from frame t to frame t+1
        b = flow_backward, from frame t+1 to frame t
        b_warp(p) = b(p + f(p))

    A pixel is marked inconsistent when:
        ||f(p) + b_warp(p)||^2
            > 0.01 * (||f(p)||^2 + ||b_warp(p)||^2) + 0.5

    The returned mask is:
        non_occlusion = consistent & in_bounds & not_motion_boundary

    Args:
        flow_forward (torch.Tensor): Flow from frame t to t+1, shaped
        (1, 2, H, W).
        flow_backward (torch.Tensor): Flow from frame t+1 to t, shaped
        (1, 2, H, W).

    Returns:
        torch.Tensor: Boolean mask shaped (1, H, W).
    """
    warped_backward = warp_tensor(flow_backward, flow_forward)
    fb_error_sq = (flow_forward + warped_backward).pow(2).sum(dim=1)
    flow_mag_sq = (
        flow_forward.pow(2).sum(dim=1)
        + warped_backward.pow(2).sum(dim=1)
    )

    consistent = fb_error_sq <= (0.01 * flow_mag_sq + 0.5)
    in_bounds = valid_coordinate_mask(flow_forward)
    not_boundary = ~motion_boundary_mask(flow_forward)
    return consistent & in_bounds & not_boundary


def compute_flow(flow_net, frame_a, frame_b, flow_max_side):
    """
    Compute optical flow from frame_a to frame_b with PTLFlow.

    Args:
        flow_net (torch.nn.Module): PTLFlow model.
        frame_a (torch.Tensor): Source frame batch shaped (B, 3, H, W).
        frame_b (torch.Tensor): Target frame batch shaped (B, 3, H, W).
        flow_max_side (int): Maximum side length used for PTLFlow inference.
            A value of 0 keeps full-resolution flow inference.

    Returns:
        torch.Tensor: Flow from frame_a to frame_b, shaped (B, 2, H, W).
        Internally, PTLFlow receives an image pair shaped (B, 2, 3, H, W).
    """
    _, _, height, width = frame_a.shape
    flow_a = frame_a
    flow_b = frame_b
    flow_height = height
    flow_width = width

    if flow_max_side > 0 and max(height, width) > flow_max_side:
        scale = float(flow_max_side) / float(max(height, width))
        flow_height = max(1, int(round(height * scale)))
        flow_width = max(1, int(round(width * scale)))
        flow_a = F.interpolate(
            frame_a,
            size=(flow_height, flow_width),
            mode="bilinear",
            align_corners=False,
        )
        flow_b = F.interpolate(
            frame_b,
            size=(flow_height, flow_width),
            mode="bilinear",
            align_corners=False,
        )

    inputs = torch.stack([flow_a, flow_b], dim=1)
    predictions = flow_net({"images": inputs})
    # PTLFlow documents predictions["flows"] as BNCHW; for a two-frame pair,
    # N is typically 1, yielding [B, 1, 2, H, W].
    flow = predictions["flows"].squeeze(1).float()

    if (flow_height, flow_width) != (height, width):
        flow = F.interpolate(
            flow,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        flow[:, 0, :, :] *= float(width) / float(flow_width)
        flow[:, 1, :, :] *= float(height) / float(flow_height)

    return flow


def compute_pair_ewarp(flow_net, frame_a, frame_b, flow_max_side):
    """
    Compute one adjacent-frame Ewarp value.

    Lai et al., "Learning Blind Video Temporal Consistency", ECCV 2018.

    Let:
        V_t = frame_a
        V_{t+1} = frame_b
        f = flow from V_t to V_{t+1}
        M_t = binary non-occlusion mask

    With V_{t+1} warped into V_t coordinates, this computes:
        Ewarp(V_t, V_{t+1})
            = sum_p M_t(p) * ||V_t(p) - warp(V_{t+1}, f)(p)||_2^2
              / sum_p M_t(p)

    Args:
        flow_net (torch.nn.Module): PTLFlow model.
        frame_a (torch.Tensor): Frame t shaped (1, 3, H, W).
        frame_b (torch.Tensor): Frame t+1 shaped (1, 3, H, W).
        flow_max_side (int): Maximum side length used for PTLFlow inference.
            A value of 0 keeps full-resolution flow inference.

    Returns:
        float or None: Raw Ewarp value, or None if no valid pixels remain.
    """
    if frame_a.shape != frame_b.shape:
        raise ValueError("frame_size_mismatch")

    with torch.inference_mode():
        flow_forward = compute_flow(
            flow_net,
            frame_a,
            frame_b,
            flow_max_side,
        )
        flow_backward = compute_flow(
            flow_net,
            frame_b,
            frame_a,
            flow_max_side,
        )
        warped_b = warp_tensor(frame_b, flow_forward)
        mask = non_occlusion_mask(flow_forward, flow_backward)
        valid_count = int(mask.sum().item())

        if valid_count == 0:
            return None

        err = (frame_a - warped_b).pow(2).sum(dim=1)
        return float(err[mask].mean().item())


def _nan_to_none(v):
    """
    Convert None, NaN, or infinite numeric values into None for strict JSON.
    """
    if v is None:
        return None

    fv = float(v)
    if not np.isfinite(fv):
        return None

    return fv


def eval_sequence_ewarp(fake_dir, flow_net, flow_max_side):
    """
    Compute average Ewarp for one generated sequence folder.

    For a sequence V with T frames:
        Ewarp(V) = (1 / (T - 1)) * sum_t Ewarp(V_t, V_{t+1})

    Args:
        fake_dir (str): Directory containing generated frame images.
        flow_net (torch.nn.Module): PTLFlow model.
        flow_max_side (int): Maximum side length used for PTLFlow inference.
            A value of 0 keeps full-resolution flow inference.

    Returns:
        dict: Sequence result with raw Ewarp and validity metadata.
    """
    frame_names = sorted(
        f for f in os.listdir(fake_dir)
        if os.path.isfile(os.path.join(fake_dir, f))
    )

    num_pairs = max(len(frame_names) - 1, 0)
    if len(frame_names) < 2:
        return {
            "ewarp_raw": None,
            "valid": False,
            "reason": "not_enough_frames",
            "num_pairs": num_pairs,
            "num_valid_pairs": 0,
        }

    pair_scores = []
    previous = load_frame(os.path.join(fake_dir, frame_names[0]), DEVICE)
    if previous is None:
        return {
            "ewarp_raw": None,
            "valid": False,
            "reason": "unreadable_frame",
            "num_pairs": num_pairs,
            "num_valid_pairs": 0,
        }

    for frame_name in frame_names[1:]:
        current = load_frame(os.path.join(fake_dir, frame_name), DEVICE)
        if current is None:
            return {
                "ewarp_raw": None,
                "valid": False,
                "reason": "unreadable_frame",
                "num_pairs": num_pairs,
                "num_valid_pairs": len(pair_scores),
            }

        try:
            pair_score = compute_pair_ewarp(
                flow_net,
                previous,
                current,
                flow_max_side,
            )
        except ValueError as e:
            if str(e) == "frame_size_mismatch":
                return {
                    "ewarp_raw": None,
                    "valid": False,
                    "reason": "frame_size_mismatch",
                    "num_pairs": num_pairs,
                    "num_valid_pairs": len(pair_scores),
                }
            raise

        if pair_score is not None:
            pair_scores.append(pair_score)

        previous = current

    if len(pair_scores) == 0:
        return {
            "ewarp_raw": None,
            "valid": False,
            "reason": "no_valid_pairs",
            "num_pairs": num_pairs,
            "num_valid_pairs": 0,
        }

    return {
        "ewarp_raw": float(np.mean(pair_scores)),
        "valid": True,
        "reason": None,
        "num_pairs": num_pairs,
        "num_valid_pairs": len(pair_scores),
    }


def _format_sequence_line(item):
    """
    Build one human-readable summary line for a sequence result.

    Args:
        item (dict): One sequence entry from the report.

    Returns:
        str: Formatted one-line sequence summary.
    """
    seq = item["sequence"]

    if not item["valid"]:
        reason = item.get("reason") or "unknown"
        return "{:>3} -> invalid ({})".format(seq, reason)

    return (
        "{:>3} -> E*warp: {:.4f}   Ewarp: {:.8f}   "
        "valid_pairs: {}/{}"
    ).format(
        seq,
        item["ewarp_scaled"],
        item["ewarp_raw"],
        item["num_valid_pairs"],
        item["num_pairs"],
    )


def compute_ewarp_report(
    fake_root,
    show_progress,
    flow_model,
    flow_ckpt,
    report_scale,
    flow_max_side,
):
    """
    Compute a structured Ewarp report over sequence folders.

    Expected layout:
      fake_root/
        000/
        001/
        ...

    Args:
        fake_root (str): Directory containing generated sequence subfolders.
        show_progress (bool): If True, print per-sequence progress lines.
        flow_model (str): PTLFlow model name.
        flow_ckpt (str): PTLFlow checkpoint name or path.
        report_scale (float): Scale applied to raw Ewarp for displayed E*warp.
        flow_max_side (int): Maximum side length used for PTLFlow inference.
            A value of 0 keeps full-resolution flow inference.

    Returns:
        dict: Report dictionary.

    Raises:
        FileNotFoundError: If fake_root does not exist or is not a directory.
        RuntimeError: If PTLFlow model creation fails.
    """
    if not os.path.isdir(fake_root):
        raise FileNotFoundError(
            "fake_root does not exist or is not a directory: {}".format(
                fake_root
            )
        )
    if flow_max_side < 0:
        raise ValueError("flow_max_side must be non-negative")

    flow_net = make_flow_model(flow_model, flow_ckpt, DEVICE)

    seq_names = sorted(os.listdir(fake_root))
    seq_dirs = []
    for s in seq_names:
        if os.path.isdir(os.path.join(fake_root, s)):
            seq_dirs.append(s)

    scored_seq_count = 0
    sequences = []
    overall_raw_vals = []

    if show_progress:
        print(
            "Scanning {} sequence folders in {}".format(
                len(seq_dirs),
                os.path.abspath(fake_root),
            ),
            flush=True,
        )

    for idx, seq in enumerate(seq_dirs, start=1):
        fake_seq = os.path.join(fake_root, seq)
        seq_result = eval_sequence_ewarp(
            fake_seq,
            flow_net,
            flow_max_side,
        )

        ewarp_raw = seq_result["ewarp_raw"]
        ewarp_scaled = (
            None
            if ewarp_raw is None
            else float(ewarp_raw * report_scale)
        )

        seq_item = {
            "sequence": str(seq),
            "ewarp_raw": _nan_to_none(ewarp_raw),
            "ewarp_scaled": _nan_to_none(ewarp_scaled),
            "valid": bool(seq_result["valid"]),
            "reason": seq_result["reason"],
            "num_pairs": int(seq_result["num_pairs"]),
            "num_valid_pairs": int(seq_result["num_valid_pairs"]),
        }

        if seq_item["valid"]:
            scored_seq_count += 1
            overall_raw_vals.append(seq_item["ewarp_raw"])

        sequences.append(seq_item)

        if show_progress:
            line = _format_sequence_line(seq_item)
            print("[{}/{}] {}".format(idx, len(seq_dirs), line), flush=True)

    overall_raw = (
        None
        if len(overall_raw_vals) == 0
        else float(np.mean(overall_raw_vals))
    )
    overall_scaled = (
        None
        if overall_raw is None
        else float(overall_raw * report_scale)
    )

    return {
        "fake_root": os.path.abspath(fake_root),
        "flow_model": flow_model,
        "flow_ckpt": flow_ckpt,
        "report_scale": float(report_scale),
        "flow_max_side": int(flow_max_side),
        "num_sequence_dirs_found": len(seq_dirs),
        "num_sequences_scored": scored_seq_count,
        "sequences": sequences,
        "overall": {
            "ewarp_raw": _nan_to_none(overall_raw),
            "ewarp_scaled": _nan_to_none(overall_scaled),
        },
    }


def print_human_report(report, include_sequences):
    """
    Print a human-readable Ewarp report.

    Args:
        report (dict): Report dictionary returned by compute_ewarp_report().
        include_sequences (bool): If True, print all per-sequence summaries.

    Returns:
        None: Prints formatted text to stdout and warnings to stderr.
    """
    if report["num_sequence_dirs_found"] == 0:
        print(
            "WARNING: No sequence subdirectories found in: {}".format(
                report["fake_root"]
            ),
            file=sys.stderr,
        )

    if include_sequences:
        for item in report["sequences"]:
            print(_format_sequence_line(item))

    print(
        "\nvalid_pairs: contributing adjacent frame pairs / "
        "total adjacent frame pairs"
    )
    print("\nOverall averages:")
    overall = report["overall"]
    if overall.get("ewarp_scaled") is None:
        print("  E*warp: no valid scores")
    else:
        print("  E*warp: {:.4f}".format(overall["ewarp_scaled"]))

    if overall.get("ewarp_raw") is None:
        print("  Ewarp: no valid scores")
    else:
        print("  Ewarp: {:.8f}".format(overall["ewarp_raw"]))


def write_json_report(report, json_out_path):
    """
    Write the metric report dictionary to a JSON file.

    Args:
        report (dict): Report dictionary returned by compute_ewarp_report().
        json_out_path (str): Output JSON file path.

    Returns:
        None: Writes the report to disk as JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(json_out_path)), exist_ok=True)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    """
    Parse command-line arguments, run Ewarp, and output results.

    Returns:
        None: Prints reports and optionally writes JSON output.
    """
    parser = argparse.ArgumentParser(
        description="Compute Ewarp over generated sequence folders."
    )
    parser.add_argument(
        "fake_root",
        help="Path to directory containing sequence subfolders.",
    )
    parser.add_argument(
        "--flow-model",
        type=str,
        default="recover_cx",
        help="PTLFlow model name (default: recover_cx).",
    )
    parser.add_argument(
        "--flow-ckpt",
        type=str,
        default="sintel",
        help="PTLFlow checkpoint name or path (default: sintel).",
    )
    parser.add_argument(
        "--report-scale",
        type=float,
        default=1000.0,
        help=(
            "Scale applied to raw Ewarp for reporting E*warp "
            "(default: 1000.0)."
        ),
    )
    parser.add_argument(
        "--flow-max-side",
        type=int,
        default=1024,
        help=(
            "Maximum side length for PTLFlow inference; 0 keeps full "
            "resolution (default: 1024)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to write a JSON report file.",
    )
    parser.add_argument(
        "--json-stdout",
        action="store_true",
        help="Also print JSON report to stdout.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable output (useful with --json-stdout).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-sequence live progress printing during evaluation.",
    )

    args = parser.parse_args()

    report = compute_ewarp_report(
        args.fake_root,
        show_progress=(not args.quiet and not args.no_progress),
        flow_model=args.flow_model,
        flow_ckpt=args.flow_ckpt,
        report_scale=args.report_scale,
        flow_max_side=args.flow_max_side,
    )

    if not args.quiet:
        print_human_report(report, include_sequences=False)

    if args.json_out:
        write_json_report(report, args.json_out)

    if args.json_stdout:
        print(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        )


if __name__ == "__main__":
    main()
