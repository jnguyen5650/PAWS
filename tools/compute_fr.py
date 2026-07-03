"""
Compute full-reference image quality metrics over paired sequence folders.

This script scans generated and ground-truth sequence subfolders, runs PSNR,
SSIM, LPIPS, and DISTS on paired frames, and builds a report with per-sequence
scores plus overall averages.

Purpose:
- Score generated outputs against matching ground-truth frames.
- Print a human-readable summary and optionally write JSON output.

Examples:
  python -m tools.compute_fr data/REDS/hr/test data/REDS/hr/gt
  python -m tools.compute_fr data/REDS/hr/test data/REDS/hr/gt \
    --json-out outputs/fr_report.json
"""

from utils.traceback import *  # noqa: F401, F403
import os
import sys
import json
import cv2
import numpy as np
import torch
import pyiqa
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_metric_models(device, lpips_net):
    """
    Create the metric models used for evaluation.

    Args:
        device (torch.device): Device used to host the metric models.
        lpips_net (str): LPIPS backbone name, either "alex" or "vgg".

    Returns:
        dict: Mapping containing initialized LPIPS and DISTS metrics.
    """
    if lpips_net == "vgg":
        lpips_name = "lpips-vgg"
    else:
        lpips_name = "lpips"

    metrics = {}
    metrics["lpips"] = pyiqa.create_metric(lpips_name, device=device)
    metrics["lpips"].eval()
    metrics["dists"] = pyiqa.create_metric("dists", device=device)
    metrics["dists"].eval()
    metrics["lpips_name"] = lpips_name
    return metrics


def load_image_for_metrics(path):
    """
    Load one image for metric evaluation.

    Args:
        path (str): Path to one image file.

    Returns:
        tuple: `(img, img_t)`, where `img` is the original OpenCV image used
        for PSNR and SSIM and `img_t` is an RGB `torch.Tensor` in [0, 1] used
        for LPIPS and DISTS. Returns `(None, None)` when the image cannot be
        read.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None

    if img.ndim == 2:
        img_rgb = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img_rgb = np.repeat(img, 3, axis=2)
    else:
        img_rgb = img

    if img_rgb.ndim == 3 and img_rgb.shape[2] > 3:
        img_rgb = img_rgb[:, :, :3]

    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img, img_t


def eval_sequence_metrics(fake_dir, gt_dir, metrics, crop_border, test_y):
    """
    Compute per-sequence metric averages.

    Args:
        fake_dir (str): Directory containing generated frames for one sequence.
        gt_dir (str): Directory containing ground-truth frames for one
            sequence.
        metrics (dict): Dictionary returned by make_metric_models.
        crop_border (int): Border size cropped for PSNR and SSIM.
        test_y (bool): If True, compute PSNR and SSIM on the Y channel.

    Returns:
        dict: Average PSNR, SSIM, LPIPS, and DISTS scores against GT frames,
        plus validity metadata.
    """
    from metrics.psnr_ssim import calculate_psnr, calculate_ssim

    fake_list = sorted(
        f for f in os.listdir(fake_dir)
        if os.path.isfile(os.path.join(fake_dir, f))
    )
    gt_list = sorted(
        f for f in os.listdir(gt_dir)
        if os.path.isfile(os.path.join(gt_dir, f))
    )

    if len(fake_list) != len(gt_list):
        return {
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "dists": None,
            "valid": False,
            "reason": "frame_count_mismatch",
        }

    if len(fake_list) == 0:
        return {
            "psnr": None,
            "ssim": None,
            "lpips": None,
            "dists": None,
            "valid": False,
            "reason": "no_frames",
        }

    psnrs = []
    ssims = []
    lpips_scores = []
    dists_scores = []

    for f_name, g_name in zip(fake_list, gt_list):
        fake_path = os.path.join(fake_dir, f_name)
        gt_path = os.path.join(gt_dir, g_name)

        fake_img, fake_t = load_image_for_metrics(fake_path)
        gt_img, gt_t = load_image_for_metrics(gt_path)

        if (
            fake_img is None
            or gt_img is None
            or fake_t is None
            or gt_t is None
        ):
            return {
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "dists": None,
                "valid": False,
                "reason": "unreadable_frame",
            }

        try:
            psnr_val = calculate_psnr(
                fake_img,
                gt_img,
                crop_border,
                input_order="HWC",
                test_y_channel=test_y,
            )
            ssim_val = calculate_ssim(
                fake_img,
                gt_img,
                crop_border,
                input_order="HWC",
                test_y_channel=test_y,
            )
        except Exception:
            return {
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "dists": None,
                "valid": False,
                "reason": "psnr_ssim_failed",
            }

        try:
            with torch.inference_mode():
                lpips_val = metrics["lpips"](fake_t, gt_t)
                dists_val = metrics["dists"](fake_t, gt_t)
        except Exception:
            return {
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "dists": None,
                "valid": False,
                "reason": "perceptual_metric_failed",
            }

        psnrs.append(float(psnr_val))
        ssims.append(float(ssim_val))
        lpips_scores.append(float(lpips_val.item()))
        dists_scores.append(float(dists_val.item()))

    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_scores)),
        "dists": float(np.mean(dists_scores)),
        "valid": True,
        "reason": None,
    }


def _format_sequence_line(item):
    """
    Build a one-line summary for a sequence result.

    Args:
        item (dict): One sequence entry from the report.

    Returns:
        str: Formatted one-line summary for console output.
    """
    seq = item["sequence"]

    if not item["valid"]:
        reason = item.get("reason") or "unknown"
        return "{:>3} -> invalid ({})".format(seq, reason)

    return (
        "{:>3} -> PSNR: {:.4f}   SSIM: {:.4f}   LPIPS: {:.4f}   "
        "DISTS: {:.4f}"
    ).format(
        seq,
        item["psnr"],
        item["ssim"],
        item["lpips"],
        item["dists"],
    )


def compute_metric_report(
    fake_root,
    gt_root,
    show_progress,
    lpips_net,
    crop_border,
    test_y,
):
    """
    Compute a structured metric report over sequence folders.

    Expected layout:
      fake_root/
        000/
        001/
        ...
      gt_root/
        000/
        001/
        ...

    Args:
        fake_root (str): Directory containing generated sequence subfolders.
        gt_root (str): Directory containing ground-truth sequence subfolders.
        show_progress (bool): If True, print per-sequence progress lines as
            they finish.
        lpips_net (str): LPIPS backbone name, either "alex" or "vgg".
        crop_border (int): Border size cropped for PSNR and SSIM.
        test_y (bool): If True, compute PSNR and SSIM on the Y channel.

    Returns:
        dict: Report dictionary with per-sequence results and overall psnr,
        ssim, lpips, and dists averages.

    Raises:
        FileNotFoundError: If fake_root or gt_root does not exist or is not a
        directory.
    """
    if not os.path.isdir(fake_root):
        raise FileNotFoundError(
            "fake_root does not exist or is not a directory: {}".format(
                fake_root
            )
        )

    if not os.path.isdir(gt_root):
        raise FileNotFoundError(
            "gt_root does not exist or is not a directory: {}".format(gt_root)
        )

    metrics = make_metric_models(DEVICE, lpips_net)

    overall_vals = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "dists": [],
    }

    seq_names = sorted(os.listdir(fake_root))
    seq_dirs = []
    for s in seq_names:
        if os.path.isdir(os.path.join(fake_root, s)):
            seq_dirs.append(s)

    sequences = []
    scored_seq_count = 0

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
        gt_seq = os.path.join(gt_root, seq)

        if not os.path.isdir(gt_seq):
            seq_item = {
                "sequence": str(seq),
                "psnr": None,
                "ssim": None,
                "lpips": None,
                "dists": None,
                "valid": False,
                "reason": "missing_gt_sequence",
            }
        else:
            seq_result = eval_sequence_metrics(
                fake_seq,
                gt_seq,
                metrics,
                crop_border,
                test_y,
            )

            seq_item = {
                "sequence": str(seq),
                "psnr": seq_result["psnr"],
                "ssim": seq_result["ssim"],
                "lpips": seq_result["lpips"],
                "dists": seq_result["dists"],
                "valid": seq_result["valid"],
                "reason": seq_result["reason"],
            }

        if seq_item["valid"]:
            scored_seq_count += 1
            overall_vals["psnr"].append(seq_item["psnr"])
            overall_vals["ssim"].append(seq_item["ssim"])
            overall_vals["lpips"].append(seq_item["lpips"])
            overall_vals["dists"].append(seq_item["dists"])

        sequences.append(seq_item)

        if show_progress:
            line = _format_sequence_line(seq_item)
            print("[{}/{}] {}".format(idx, len(seq_dirs), line), flush=True)

    overall = {}
    for name in ["psnr", "ssim", "lpips", "dists"]:
        vals = overall_vals[name]
        if len(vals) == 0:
            overall[name] = None
        else:
            overall[name] = float(np.mean(vals))

    report = {
        "fake_root": os.path.abspath(fake_root),
        "gt_root": os.path.abspath(gt_root),
        "lpips_model": metrics["lpips_name"],
        "crop_border": int(crop_border),
        "test_y": bool(test_y),
        "num_sequence_dirs_found": len(seq_dirs),
        "num_sequences_scored": scored_seq_count,
        "sequences": sequences,
        "overall": {
            "psnr": overall["psnr"],
            "ssim": overall["ssim"],
            "lpips": overall["lpips"],
            "dists": overall["dists"],
        },
    }
    return report


def print_human_report(report, include_sequences):
    """
    Print a human-readable report for reference metrics.

    Args:
        report (dict): Report dictionary returned by compute_metric_report.
        include_sequences (bool): If True, print per-sequence lines from the
            report.

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

    print("\nOverall averages:")
    overall = report["overall"]

    if overall.get("psnr") is None:
        print("  PSNR: no valid scores")
    else:
        print("  PSNR: {:.4f}".format(overall["psnr"]))

    if overall.get("ssim") is None:
        print("  SSIM: no valid scores")
    else:
        print("  SSIM: {:.4f}".format(overall["ssim"]))

    if overall.get("lpips") is None:
        print("  LPIPS: no valid scores")
    else:
        print("  LPIPS: {:.4f}".format(overall["lpips"]))

    if overall.get("dists") is None:
        print("  DISTS: no valid scores")
    else:
        print("  DISTS: {:.4f}".format(overall["dists"]))


def write_json_report(report, json_out_path):
    """
    Write the report dictionary to a JSON file.

    Args:
        report (dict): Report dictionary returned by compute_metric_report.
        json_out_path (str): Output JSON file path.

    Returns:
        None: Writes the report to disk as JSON.
    """
    os.makedirs(os.path.dirname(os.path.abspath(json_out_path)), exist_ok=True)
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    """
    Parse command-line arguments, run reference metrics, and output results.

    Returns:
        None: Prints reports and optionally writes JSON output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute reference IQA metrics (PSNR, SSIM, LPIPS, DISTS) over "
            "paired sequence folders."
        )
    )
    parser.add_argument(
        "fake_root",
        type=str,
        help="Path to directory containing generated sequence subfolders.",
    )
    parser.add_argument(
        "gt_root",
        type=str,
        help="Path to directory containing ground-truth sequence subfolders.",
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
        help="Also print the JSON report to stdout.",
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
    parser.add_argument(
        "--lpips-net",
        type=str,
        choices=("alex", "vgg"),
        default="alex",
        help="LPIPS backbone to use via pyiqa (default: alex).",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Crop border used for PSNR and SSIM (default: 0).",
    )
    parser.add_argument(
        "--test-y",
        action="store_true",
        help="Compute PSNR and SSIM on the Y channel.",
    )

    args = parser.parse_args()

    report = compute_metric_report(
        args.fake_root,
        args.gt_root,
        show_progress=(not args.quiet and not args.no_progress),
        lpips_net=args.lpips_net,
        crop_border=args.crop_border,
        test_y=args.test_y,
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
