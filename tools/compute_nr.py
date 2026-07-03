"""
Compute no-reference image quality metrics over sequence folders.

This script scans a directory of sequence subfolders, runs NIQE, MUSIQ,
CLIPIQA, and COVER on the generated frames, and builds a report with
per-sequence scores plus overall averages.

Purpose:
- Score generated outputs without requiring ground-truth frames.
- Print a human-readable summary and optionally write JSON output.

Examples:
  python -m tools.compute_nr data/REDS/hr/test
  python -m tools.compute_nr data/REDS/hr/test --json-out outputs/nr_report.json
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


def make_metric_models(device):
    """
    Create the metric models used for evaluation.

    Args:
        device (torch.device): Device used to host the metric models.

    Returns:
        dict: Mapping of metric name to initialized pyiqa metric object.
    """
    metric_names = ["niqe", "musiq", "clipiqa"]
    metrics = {}
    for name in metric_names:
        metric = pyiqa.create_metric(name, device=device)
        metric.eval()
        metrics[name] = metric
    return metrics


def load_image_for_metrics(path):
    """
    Load one image for metric evaluation.

    Args:
        path (str): Path to one image file.

    Returns:
        torch.Tensor or None: RGB tensor shaped (1, 3, H, W) in [0, 1], or None
        when the image is unreadable or too small for NIQE.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Handle grayscale or single-channel images
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    # If image has alpha or more channels, keep only first 3
    if img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]

    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # NIQE uses patches and can fail on very small images
    if h < 96 or w < 96:
        return None

    img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return img_t


def eval_sequence_metrics(fake_dir, metrics):
    """
    Compute per-sequence metric averages.

    Args:
        fake_dir (str): Directory containing generated frame images for one
            sequence.
        metrics (dict): Mapping of metric name to initialized pyiqa metric
            object.

    Returns:
        dict: Average NIQE, MUSIQ, and CLIPIQA scores. Metrics with no valid
        frames are reported as np.nan.
    """
    fake_list = sorted(os.listdir(fake_dir))

    scores = {name: [] for name in metrics.keys()}

    for f_name in fake_list:
        f_path = os.path.join(fake_dir, f_name)
        if not os.path.isfile(f_path):
            continue

        img_t = load_image_for_metrics(f_path)
        if img_t is None:
            continue

        with torch.no_grad():
            for name, metric in metrics.items():
                try:
                    score = metric(img_t)
                    scores[name].append(float(score.item()))
                except Exception:
                    continue

    avg_scores = {}
    for name, vals in scores.items():
        if len(vals) == 0:
            avg_scores[name] = np.nan
        else:
            avg_scores[name] = float(np.mean(vals))

    return avg_scores


def eval_sequence_cover(fake_dir, cover_metric):
    """
    Compute one COVER overall score for a sequence folder.

    Args:
        fake_dir (str): Directory containing generated frame images for one
            sequence.
        cover_metric (CoverMetric): Initialized COVER metric wrapper.

    Returns:
        float: COVER overall score, or np.nan if scoring fails.
    """
    try:
        return float(cover_metric(fake_dir))
    except Exception:
        return np.nan


def _nan_to_none(v):
    """
    Convert None, NaN, or infinite numeric values into None for JSON output.

    Args:
        v (object): None or a numeric value.

    Returns:
        float or None: Finite float value, or None for strict JSON.
    """
    if v is None:
        return None

    fv = float(v)
    if not np.isfinite(fv):
        return None

    return fv


def _format_sequence_line(item):
    """
    Build one human-readable summary line for a sequence result.

    Args:
        item (dict): One sequence entry from the report.

    Returns:
        str: Formatted one-line summary for console output.
    """
    seq = item["sequence"]

    if not item["valid"]:
        return "{:>3} -> no valid scores".format(seq)

    parts = []

    if item["niqe"] is not None:
        parts.append("NIQE: {:.4f}".format(item["niqe"]))
    if item["musiq"] is not None:
        parts.append("MUSIQ: {:.4f}".format(item["musiq"]))
    if item["clipiqa"] is not None:
        parts.append("CLIPIQA: {:.4f}".format(item["clipiqa"]))
    if item["cover"] is not None:
        parts.append("COVER: {:.4f}".format(item["cover"]))

    return "{:>3} -> {}".format(seq, "   ".join(parts))


def compute_metric_report(fake_root, show_progress):
    """
    Compute a structured metric report over sequence folders.

    Expected layout:
      fake_root/
        000/
        001/
        ...

    Args:
        fake_root (str): Directory containing generated sequence subfolders.
        show_progress (bool): If True, print per-sequence progress lines as
            each sequence finishes.

    Returns:
        dict: Report dictionary with per-sequence results and overall niqe,
        musiq, clipiqa, and cover averages.

    Raises:
        FileNotFoundError: If fake_root does not exist or is not a directory.
    """
    if not os.path.isdir(fake_root):
        raise FileNotFoundError(
            "fake_root does not exist or is not a directory: {}".format(
                fake_root
            )
        )

    metrics = make_metric_models(DEVICE)
    from metrics.cover_metric import CoverMetric

    cover_metric = CoverMetric(device=DEVICE)

    overall_vals = {
        "niqe": [],
        "musiq": [],
        "clipiqa": [],
        "cover": [],
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

        seq_scores = eval_sequence_metrics(fake_seq, metrics)
        cover_val = eval_sequence_cover(fake_seq, cover_metric)

        niqe_val = seq_scores.get("niqe", np.nan)
        musiq_val = seq_scores.get("musiq", np.nan)
        clip_val = seq_scores.get("clipiqa", np.nan)

        all_nan = all(
            np.isnan(v)
            for v in [niqe_val, musiq_val, clip_val, cover_val]
        )

        if not all_nan:
            scored_seq_count += 1

            if not np.isnan(niqe_val):
                overall_vals["niqe"].append(niqe_val)
            if not np.isnan(musiq_val):
                overall_vals["musiq"].append(musiq_val)
            if not np.isnan(clip_val):
                overall_vals["clipiqa"].append(clip_val)
            if not np.isnan(cover_val):
                overall_vals["cover"].append(cover_val)

        seq_item = {
            "sequence": str(seq),
            "niqe": _nan_to_none(niqe_val),
            "musiq": _nan_to_none(musiq_val),
            "clipiqa": _nan_to_none(clip_val),
            "cover": _nan_to_none(cover_val),
            "valid": not all_nan,
        }

        sequences.append(seq_item)

        if show_progress:
            line = _format_sequence_line(seq_item)
            print("[{}/{}] {}".format(idx, len(seq_dirs), line), flush=True)

    overall = {}
    for name in ["niqe", "musiq", "clipiqa", "cover"]:
        vals = overall_vals[name]
        if len(vals) == 0:
            overall[name] = None
        else:
            overall[name] = float(np.mean(vals))

    report = {
        "fake_root": os.path.abspath(fake_root),
        "num_sequence_dirs_found": len(seq_dirs),
        "num_sequences_scored": scored_seq_count,
        "sequences": sequences,
        "overall": {
            "niqe": _nan_to_none(overall["niqe"]),
            "musiq": _nan_to_none(overall["musiq"]),
            "clipiqa": _nan_to_none(overall["clipiqa"]),
            "cover": _nan_to_none(overall["cover"]),
        },
    }
    return report


def print_human_report(report, include_sequences):
    """
    Print a human-readable report for metric scores.

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
    for name in ["niqe", "musiq", "clipiqa", "cover"]:
        val = overall.get(name, None)
        if val is None:
            print("  {}: no valid scores".format(name.upper()))
        else:
            print("  {}: {:.4f}".format(name.upper(), val))


def write_json_report(report, json_out_path):
    """
    Write the metric report dictionary to a JSON file.

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
    Parse command-line arguments, run no-reference metrics, and output results.

    Returns:
        None: Prints reports and optionally writes JSON output.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute no-reference IQA metrics (NIQE, MUSIQ, CLIPIQA) and "
            "COVER over sequence folders."
        )
    )
    parser.add_argument(
        "fake_root",
        help="Path to directory containing sequence subfolders.",
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

    report = compute_metric_report(
        args.fake_root,
        show_progress=(not args.quiet and not args.no_progress),
    )

    if not args.quiet:
        # Per-sequence lines were already streamed during
        # compute_metric_report().
        # Print only the final summary here to avoid duplicate sequence output.
        print_human_report(report, include_sequences=False)

    if args.json_out:
        write_json_report(report, args.json_out)

    if args.json_stdout:
        print(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        )


if __name__ == "__main__":
    main()
