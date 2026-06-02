"""
Select the best-ranked checkpoint with coarse-to-fine metric-guided search.

Evaluating every saved checkpoint linearly is expensive because each candidate
requires extracting weights, running video super-resolution inference, and
computing quality metrics. This script reduces that cost by evaluating a
coarse epoch grid first, ranking evaluated checkpoints with either a weighted
objective or a Pareto-frontier objective, then refining the next candidate grid
around the best checkpoints found so far.

For each candidate epoch, the script extracts generator weights, runs test.py,
scores the generated sequence outputs with no-reference or full-reference
metrics, and stores raw metrics plus ranking state in a resumable JSON file.

Purpose:
- Avoid exhaustive linear checkpoint sweeps when many epochs are saved.
- Focus expensive inference and metric evaluations near promising epochs.
- Support weighted or Pareto checkpoint ranking over NR or FR metric sets.
- Resume interrupted searches without re-evaluating cached epochs.

Usage:
  Edit the configuration constants near the top of this file, then run:
    python -m tools.search_best_epoch
"""

from dataclasses import dataclass, field
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Constant representing negative infinity, used as a default score for unranked epochs
NEG_INF = float("-inf")


# ============================================================
# SEARCH / OBJECTIVE CONFIGURATION
# ============================================================
# "weighted" = weighted sum of metrics
# "pareto" = keep only non-dominated checkpoints, then pick one
# frontier checkpoint by weighted distance to the ideal point.
OBJECTIVE_MODE = "pareto"
# "nr" = no-reference metrics
# "fr" = full-reference metrics that require GT_DIR.
METRIC_MODE = "nr"
EPOCH_MIN = 1  # First epoch included in the search range.
EPOCH_MAX = 1200  # Last epoch included in the search range.
INITIAL_STEP = 300  # Coarse stride used for the first search round.
TOPK = 1  # Number of best epochs that seed the next refinement round.
ROUND_MAX = 10  # Maximum number of coarse-to-fine refinement rounds.

# ============================================================
# TILE / PADDING CONFIGURATION (for inference)
# ============================================================
TILE_T = 0  # Temporal tile size passed to inference.
TILE_H = 256  # Tile height used during inference.
TILE_W = 256  # Tile width used during inference.
PAD_T = 0  # Temporal overlap padding around each tile.
PAD_H = 20  # Vertical overlap padding around each tile.
PAD_W = 20  # Horizontal overlap padding around each tile.

# ============================================================
# PATHS
# ============================================================
# Directory that holds checkpoint_epoch_*.pth files.
CKPT_DIR = "checkpoints"
# Inference config passed into test.py.
CFG = "configs/PAWS_RealBasicVSRPP_HAT_Stage1_PSNR_REDSx4.yaml"
# Extracted generator weights written before inference.
OUT_PTH = "outputs/Temp_G_EXTRACTED.pth"
# Output directory scored by the metric script after inference.
SRC_DIR = "data/REDS/hr/test"
GT_DIR = ""  # Ground-truth root used only when METRIC_MODE is "fr".
# Resume cache for raw metrics and cached rankings.
STATE_FILE = "outputs/checkpoint_search_state.json"

# ============================================================
# LOGGING / SEARCH BEHAVIOR
# ============================================================
# 1 = print progress and stage timing; 0 = stay quiet except for errors.
VERBOSE = 1
# 1 = print round candidates; 0 = skip candidate list logging.
SHOW_CANDIDATE_LIST = 1
# Preview limit before candidate logging switches to head/tail view.
CANDIDATE_LIST_MAX = 40
# 1 = add a shifted candidate grid; 0 = search only the base step grid.
USE_OFFSET_GRID = 1
# Number of early rounds that use the shifted candidate grid.
OFFSET_GRID_ROUNDS = 1
# Shift amount; 0 means auto half-step offset for the current round.
OFFSET_VALUE = 0

# ============================================================
# WEIGHTED OBJECTIVE WEIGHTS (for OBJECTIVE_MODE="weighted")
# ============================================================
# Negative weights mean lower values are better (e.g., NIQE, LPIPS, DISTS)
# Positive weights mean higher values are better (e.g., MUSIQ, CLIPIQA, COVER, PSNR, SSIM)
W_NIQE = -1.0
W_MUSIQ = 1.0
W_CLIPIQA = 1.0
W_COVER = 1.0
W_PSNR = 1.0
W_SSIM = 1.0
W_LPIPS = -1.0
W_DISTS = -1.0

# ============================================================
# PARETO DISTANCE WEIGHTS (for OBJECTIVE_MODE="pareto")
# ============================================================
# All metrics weighted equally for Pareto distance calculation to ideal point
P_W_NIQE = 1.0
P_W_MUSIQ = 1.0
P_W_CLIPIQA = 1.0
P_W_COVER = 1.0
P_W_PSNR = 1.0
P_W_SSIM = 1.0
P_W_LPIPS = 1.0
P_W_DISTS = 1.0

# ============================================================
# FULL-REFERENCE OPTIONS (for METRIC_MODE="fr")
# ============================================================
# "alex" or "vgg" LPIPS backbone used by the FR metric script.
FR_LPIPS_NET = "alex"
FR_CROP_BORDER = 0  # Border cropped before PSNR/SSIM calculation in FR mode.
FR_TEST_Y = 0  # 1 = evaluate PSNR/SSIM on Y (luminance) only; 0 = use full RGB images

# ============================================================
# SCRIPT / MODULE NAMES
# ============================================================
# Module that extracts generator weights from each checkpoint.
EXTRACT_MODULE = "tools.extract_submodel"
# Inference script run after extracting generator weights.
TEST_SCRIPT = "test.py"
# "fp32", "bf16", or "fp16" precision passed to inference.
TEST_PRECISION = "fp32"
# "none", "default", "reduce-overhead", or "max-autotune"
# compile mode for inference (torch.compile options)
TEST_COMPILE_MODE = "none"
# Metric module used when METRIC_MODE is "nr" (no-reference)
COMPUTE_NR_MODULE = "tools.compute_nr"
# Metric module used when METRIC_MODE is "fr" (full-reference)
COMPUTE_FR_MODULE = "tools.compute_fr"


@dataclass
class SearchState:
    """
    Mutable state for one checkpoint search run.

    Stores the active metric configuration, cached epoch results, current
    best-so-far fields, and the refinement round currently being processed.
    """
    mode: str  # Metric family: "nr" or "fr".

    # Active metric keys in scoring/logging order.
    # NR example: ["niqe", "musiq", "clipiqa", "cover"]
    # FR example: ["psnr", "ssim", "lpips", "dists"]
    metrics: list[str]
    directions: dict[str, str]  # Same keys as metrics; values are "min" or "max".
    weights: dict[str, float]  # Same keys as metrics; weighted-objective coefficients.
    pareto_weights: dict[str, float]  # Same keys as metrics; Pareto distance coefficients.
    metric_module: str  # Module run by python -m to compute metrics for mode.
    
    objective: str  # Ranking mode: "weighted" or "pareto".

    # Epoch number -> metric row and derived ranking fields.
    # Example: {300: {"metrics": {"niqe": 4.2, ...}, "score": -0.3, "pareto": 1}}
    results: dict = field(default_factory=dict)
    # Best ranked epoch and score after recompute_scores().
    # Higher score is always better: weighted mode uses the weighted metric sum;
    # Pareto mode uses negative distance to the ideal point, so closer to 0 wins.
    best_epoch: int | None = None
    best_score: float | None = None

    # Metric snapshot for best_epoch, using the same keys as metrics.
    # Example: {"niqe": 4.2, "musiq": 71.0, "clipiqa": 0.68, "cover": 0.91}
    best_metrics: dict | None = None
    current_round: int = 0


def epoch_rank_key(search_state, epoch):
    """
    Build the stable ranking key for one evaluated epoch.

    The key sorts by score first, then metric tie-breakers in their active
    directions, then the epoch number.

    Args:
        search_state (SearchState): State containing results and directions.
        epoch (int): Epoch number to rank.

    Returns:
        tuple: Sort key suitable for selecting the best epoch with min().
    """
    row = search_state.results[epoch]

    # Higher score is better, so negate for min() to work correctly
    key = [-row["score"]]

    # Add metric values as tie-breakers in their active directions
    for name in search_state.metrics:
        value = row["metrics"][name]
        # For "min" metrics, lower is better (keep positive); for "max" metrics,
        # higher is better (negate so min() selects higher values)
        key.append(value if search_state.directions[name] == "min" else -value)

    # Final tie-breaker: lower epoch number wins
    key.append(epoch)
    return tuple(key)


def format_metrics(metric_names, metrics, decimals):
    """
    Format a metrics dict into one log-friendly string.

    Args:
        metric_names (list[str]): Metric names to print in order.
        metrics (dict or None): Metric values keyed by name.
        decimals (int): Decimal places for finite values.

    Returns:
        str: Joined metric string like `NIQE=5.1234 MUSIQ=70.0000`.
    """
    parts = []
    for name in metric_names:
        try:
            value = (
                float(metrics.get(name))
                if metrics is not None
                else float("nan")
            )
        except (AttributeError, TypeError, ValueError):
            value = float("nan")

        # Format finite values with specified precision; show NaN for invalid values
        if math.isfinite(value):
            text = f"{value:.{decimals}f}"
        else:
            text = "NaN"

        parts.append(f"{name.upper()}={text}")

    return " ".join(parts)


def log(msg):
    """
    Log a message with timestamp if VERBOSE is enabled.

    Args:
        msg (str): Message to log.
    """
    if VERBOSE:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)


def log_round_candidates(candidates):
    """
    Print the remaining candidate list during the current round.

    Args:
        candidates (list[int]): Remaining candidate epochs.
    """
    if int(SHOW_CANDIDATE_LIST) != 1 or not candidates:
        return

    max_show = int(CANDIDATE_LIST_MAX)

    # If candidates fit within the limit, show all
    if len(candidates) <= max_show:
        candidate_text = " ".join(str(epoch) for epoch in candidates)
        log(f"Remaining candidates: {candidate_text}")
        return

    # Otherwise show head and tail with ellipsis in between
    head_count = max_show // 2
    tail_count = max_show - head_count

    head = " ".join(str(epoch) for epoch in candidates[:head_count])
    tail = " ".join(str(epoch) for epoch in candidates[-tail_count:])

    log(f"Remaining candidates (preview): {head} ... {tail}")


def log_round_summary(search_state, top_epochs):
    """
    Print the round summary, including best rows and best overall state.

    Args:
        search_state (SearchState): Search state with scored results.
        top_epochs (list[int]): Best epochs produced by the round.
    """
    print_section_banner(f"Top {TOPK} this round")
    for epoch in top_epochs:
        row = search_state.results[epoch]
        log(
            f"  {row['score']:.6f} {epoch} "
            f"({format_metrics(search_state.metrics, row['metrics'], 6)})"
        )

    # For Pareto mode, report the size of the current Pareto frontier
    if search_state.objective == "pareto":
        frontier_size = sum(
            1 for row in search_state.results.values()
            if row["pareto"] == 1
        )
        print_section_banner(f"Current Pareto frontier size: {frontier_size}")

    # Report the overall best epoch found so far across all rounds
    if search_state.best_epoch is not None:
        print_section_banner(
            f"Best overall so far: epoch={search_state.best_epoch} "
            f"score={search_state.best_score:.6f} "
            f"({format_metrics(search_state.metrics, search_state.best_metrics, 6)})"
        )


def print_banner(title):
    """
    Print a section banner through the normal logging helper.

    Args:
        title (str): Banner title text.
    """
    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


def print_section_banner(title):
    """
    Print a lightweight section separator through the normal logging helper.

    Args:
        title (str): Section title text.
    """
    log("")
    log("-" * 72)
    log(title)
    log("-" * 72)


def validate_settings():
    """
    Validate the editable top-of-file settings before running the search.

    Returns:
        None: Settings are valid.

    Raises:
        ValueError: If a setting is invalid or a configured path is missing.
    """
    objective = str(OBJECTIVE_MODE).lower()
    metric_mode = str(METRIC_MODE).lower()
    precision = str(TEST_PRECISION).lower()
    compile_mode = str(TEST_COMPILE_MODE).lower()

    if objective not in ("weighted", "pareto"):
        raise ValueError("OBJECTIVE_MODE must be 'weighted' or 'pareto'.")
    if metric_mode not in ("nr", "fr"):
        raise ValueError("METRIC_MODE must be 'nr' or 'fr'.")
    if precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("TEST_PRECISION must be 'fp32', 'bf16', or 'fp16'.")
    if compile_mode not in (
        "none",
        "default",
        "reduce-overhead",
        "max-autotune",
    ):
        raise ValueError(
            "TEST_COMPILE_MODE must be 'none', 'default', "
            "'reduce-overhead', or 'max-autotune'."
        )
    if int(EPOCH_MIN) > int(EPOCH_MAX):
        raise ValueError("EPOCH_MIN must be less than or equal to EPOCH_MAX.")
    if int(INITIAL_STEP) <= 0:
        raise ValueError("INITIAL_STEP must be a positive integer.")
    if int(TOPK) <= 0:
        raise ValueError("TOPK must be a positive integer.")
    if int(ROUND_MAX) <= 0:
        raise ValueError("ROUND_MAX must be a positive integer.")

    for name, value in (
        ("TILE_T", TILE_T),
        ("TILE_H", TILE_H),
        ("TILE_W", TILE_W),
        ("PAD_T", PAD_T),
        ("PAD_H", PAD_H),
        ("PAD_W", PAD_W),
    ):
        if int(value) < 0:
            raise ValueError(f"{name} must be a non-negative integer.")

    if int(OFFSET_GRID_ROUNDS) < 0:
        raise ValueError("OFFSET_GRID_ROUNDS must be a non-negative integer.")

    if not Path(CKPT_DIR).is_dir():
        raise ValueError(f"CKPT_DIR does not exist or is not a directory: {CKPT_DIR}")
    if not Path(CFG).is_file():
        raise ValueError(f"CFG does not exist or is not a file: {CFG}")
    if not Path(TEST_SCRIPT).is_file():
        raise ValueError(f"TEST_SCRIPT does not exist or is not a file: {TEST_SCRIPT}")
    if not Path(SRC_DIR).is_dir():
        raise ValueError(f"SRC_DIR does not exist or is not a directory: {SRC_DIR}")

    if metric_mode == "fr":
        if not str(GT_DIR).strip():
            raise ValueError("GT_DIR must be set when METRIC_MODE='fr'.")
        if not Path(GT_DIR).is_dir():
            raise ValueError(f"GT_DIR does not exist or is not a directory: {GT_DIR}")
        if str(FR_LPIPS_NET).lower() not in ("alex", "vgg"):
            raise ValueError("FR_LPIPS_NET must be 'alex' or 'vgg'.")

    return None


def initialize_search_state():
    """
    Create the initial search state from the active metric mode.

    NR mode enables no-reference metrics and weights. FR mode enables
    full-reference metrics and weights. The returned state starts with no
    cached epoch results.

    Returns:
        SearchState: State configured for the current top-of-file settings.

    Raises:
        ValueError: If METRIC_MODE is not supported.
    """
    if METRIC_MODE == "nr":
        metrics = ["niqe", "musiq", "clipiqa", "cover"]
        directions = {
            "niqe": "min",
            "musiq": "max",
            "clipiqa": "max",
            "cover": "max",
        }
        weights = {
            "niqe": W_NIQE,
            "musiq": W_MUSIQ,
            "clipiqa": W_CLIPIQA,
            "cover": W_COVER,
        }
        pareto_weights = {
            "niqe": P_W_NIQE,
            "musiq": P_W_MUSIQ,
            "clipiqa": P_W_CLIPIQA,
            "cover": P_W_COVER,
        }
        metric_module = COMPUTE_NR_MODULE
    elif METRIC_MODE == "fr":
        metrics = ["psnr", "ssim", "lpips", "dists"]
        directions = {
            "psnr": "max",
            "ssim": "max",
            "lpips": "min",
            "dists": "min",
        }
        weights = {
            "psnr": W_PSNR,
            "ssim": W_SSIM,
            "lpips": W_LPIPS,
            "dists": W_DISTS,
        }
        pareto_weights = {
            "psnr": P_W_PSNR,
            "ssim": P_W_SSIM,
            "lpips": P_W_LPIPS,
            "dists": P_W_DISTS,
        }
        metric_module = COMPUTE_FR_MODULE
    else:
        raise ValueError(f"Unknown METRIC_MODE='{METRIC_MODE}'. Expected 'nr' or 'fr'.")
    
    state = SearchState(
        mode=METRIC_MODE,
        metrics=metrics,
        directions=directions,
        weights=weights,
        pareto_weights=pareto_weights,
        metric_module=metric_module,
        objective=OBJECTIVE_MODE.lower(),
    )
    
    return state


def dominates(search_state, left_epoch, right_epoch):
    """
    Check whether one epoch Pareto-dominates another.

    Args:
        search_state (SearchState): Search state with evaluated metrics.
        left_epoch (int): Candidate dominating epoch.
        right_epoch (int): Candidate dominated epoch.

    Returns:
        bool: True when `left_epoch` is at least as good on every
        metric and better on one.
    """
    left = search_state.results[left_epoch]["metrics"]
    right = search_state.results[right_epoch]["metrics"]
    strictly_better = False

    for name in search_state.metrics:
        left_value = left[name]
        right_value = right[name]

        if search_state.directions[name] == "min":
            # For "min" metrics, lower is better
            if left_value > right_value:
                return False # left is worse
            if left_value < right_value:
                strictly_better = True # left is strictly better
        else:
            # For "max" metrics, higher is better
            if left_value < right_value:
                return False
            if left_value > right_value:
                strictly_better = True

    return strictly_better


def recompute_scores(search_state):
    """
    Recompute ranking fields for all epochs in the search state.

    Weighted mode recomputes direct weighted scores. Pareto mode rebuilds the
    frontier and scores frontier epochs by distance to the ideal point.

    Args:
        search_state (SearchState): The search state with raw metric results.

    Returns:
        None: The function updates the search_state in place.
    """
    # Reset best tracking and scoring fields
    search_state.best_epoch = None
    search_state.best_score = None
    search_state.best_metrics = None
    
    # Reset all epochs to unranked state
    for row in search_state.results.values():
        row["score"] = NEG_INF
        row["pareto"] = 0
    
    # Identify valid epochs that have finite metric values for all active metrics.
    # Only these epochs will be considered for scoring and Pareto frontier calculation.
    valid_epochs = []
    for epoch, row in search_state.results.items():
        metrics = row["metrics"]
        is_valid = True

        for name in search_state.metrics:
            try:
                value = float(metrics.get(name, float("nan")))
            except (AttributeError, TypeError, ValueError):
                is_valid = False
                break

            if not math.isfinite(value):
                is_valid = False
                break

        if is_valid:
            valid_epochs.append(epoch)
    
    # No valid epochs exist, thus nothing to score
    if not valid_epochs:
        return
    
    # Compute scores based on the selected objective mode.
    if search_state.objective == "weighted":
        # For weighted scoring, we compute a weighted sum of the metrics for each valid epoch.
        # Higher score is better (weights already account for min/max direction).
        for epoch in valid_epochs:
            metrics = search_state.results[epoch]["metrics"]
            
            score = 0.0
            for name in search_state.metrics:
                # Weighted sum: positive weights for "max" metrics, negative for "min"
                score += search_state.weights[name] * metrics[name]

            search_state.results[epoch]["score"] = score
    else:
        # For Pareto scoring, we first compute the range of each metric across valid epochs 
        # to enable normalized distance calculations.
        metric_ranges = {}
        for name in search_state.metrics:
            values = [
                search_state.results[epoch]["metrics"][name]
                for epoch in valid_epochs
            ]
            low = min(values)
            high = max(values)
            span = high - low
            if span == 0:
                # Flat metrics should add zero normalized badness instead of
                # triggering a divide-by-zero. We set span to 1.0 so normalized = 0.
                span = 1.0
            
            metric_ranges[name] = (low, high, span)
        
        # Mark epochs that are not dominated by any other valid epoch.
        for epoch in valid_epochs:
            dominated = any(
                other != epoch and dominates(search_state, other, epoch)
                for other in valid_epochs
            )
            if not dominated:
                search_state.results[epoch]["pareto"] = 1
        
        # Among the Pareto frontier epochs, compute a weighted negative distance to the ideal point.
        # The ideal point represents the best possible value for each metric (min or max).
        for epoch in valid_epochs:
            if search_state.results[epoch]["pareto"] == 1:
                metrics = search_state.results[epoch]["metrics"]
                distance_squared = 0.0
                
                for name in search_state.metrics:
                    value = metrics[name]
                    low, high, span = metric_ranges[name]
                    
                    # Normalize to badness in [0, 1]: 0 is ideal, 1 is worst observed.
                    if search_state.directions[name] == "min":
                        # For "min" metrics, the ideal is the lowest value (low).
                        # Normalized = 0 at the ideal (low), increases as value gets worse (higher).
                        # Formula: (value - low) / span gives 0 at ideal, 1 at worst.
                        normalized = (value - low) / span
                    else:
                        # For "max" metrics, the ideal is the highest value (high).
                        # Normalized = 0 at the ideal (high), increases as value gets worse (lower).
                        # Formula: (high - value) / span gives 0 at ideal, 1 at worst.
                        normalized = (high - value) / span
                    
                    weight = search_state.pareto_weights[name]
                    distance_squared += weight * normalized * normalized
                
                # Score is negative distance (higher score = closer to ideal = better)
                search_state.results[epoch]["score"] = -math.sqrt(distance_squared)
    
    # Find the best epoch using the ranking key (higher score, then better metrics, then lower epoch)
    best_epoch = min(valid_epochs, key=lambda epoch: epoch_rank_key(search_state, epoch))
    search_state.best_epoch = best_epoch
    search_state.best_score = search_state.results[best_epoch]["score"]
    search_state.best_metrics = search_state.results[best_epoch]["metrics"] 


def load_search_state(search_state):
    """
    Load cached checkpoint search results into an existing search state.

    The state file stores raw metric rows plus derived ranking fields. Raw
    metrics can be reused after weight or objective changes, but cached score,
    Pareto, and best-summary fields are only valid when the active objective,
    weights, and pareto_weights still match.

    Args:
        search_state (SearchState): State object to populate from STATE_FILE.

    Returns:
        SearchState: The same state object passed by the caller.

    Raises:
        RuntimeError: If the cached metric mode, metric list, or evaluated
        epoch rows are incompatible with the active search settings.
    """
    state_path = Path(STATE_FILE)

    # No resume cache yet; start from the empty state.
    if not state_path.is_file():
        search_state.results = {}
        return search_state

    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    saved_mode = str(data.get("metric_mode", ""))
    saved_metrics = list(data.get("metrics", []))

    # Validate that cached state matches current configuration
    if saved_mode != search_state.mode:
        raise RuntimeError(
            f"Metric mode in state file ('{saved_mode}') does not match "
            f"expected mode ('{search_state.mode}')."
        )
    if saved_metrics != search_state.metrics:
        raise RuntimeError(
            f"Metrics in state file ({saved_metrics}) do not match "
            f"expected metrics ({search_state.metrics})."
        )

    # Cached ranking fields depend on the objective and both weight sets.
    # If any of these changed, we need to recompute scores.
    settings_match = (
        data.get("objective_mode", "").lower() == search_state.objective
        and data.get("weights", {}) == search_state.weights
        and data.get("pareto_weights", {}) == search_state.pareto_weights
    )

    search_state.results = {}
    evaluated_epochs = data.get("evaluated_epochs", {})

    if not isinstance(evaluated_epochs, dict):
        raise RuntimeError(f"Invalid evaluated_epochs in state file: {STATE_FILE}")

    # Restore raw metrics for every evaluated epoch.
    for epoch_text, row in evaluated_epochs.items():
        metrics = {}
        for name in search_state.metrics:
            if name not in row:
                raise RuntimeError(
                    f"Metric '{name}' missing for epoch {epoch_text}"
                )
            metrics[name] = float(row[name])

        # Cached score and Pareto values are derived from objective and weight
        # settings. If those settings changed, keep placeholders until recompute.
        score = NEG_INF
        pareto = 0
        if settings_match:
            if row.get("score") is not None:
                score = float(row["score"])
            if "pareto" in row:
                pareto = int(row["pareto"])

        search_state.results[int(epoch_text)] = {
            "metrics": metrics,
            "score": score,
            "pareto": pareto,
        }

    # Try to restore best epoch summary from cache
    best_epoch = data.get("best_epoch")
    best_score = data.get("best_score")
    best_metrics_data = data.get("best_metrics")

    # The best summary is derived from the same ranking fields. Only restore it
    # when the per-epoch derived fields are valid for the active settings.
    can_restore_best = (
        settings_match
        and best_epoch is not None
        and int(best_epoch) in search_state.results
        and best_score is not None
    )

    if can_restore_best:
        search_state.best_epoch = int(best_epoch)
        search_state.best_score = float(best_score)
        search_state.best_metrics = {}

        # Restore the best metric snapshot. If even one required metric is
        # missing, the cached best block is incomplete, so recompute instead.
        for name in search_state.metrics:
            if name not in best_metrics_data:
                recompute_scores(search_state)
                return search_state
            search_state.best_metrics[name] = float(best_metrics_data[name])
    else:
        # Settings changed or cache incomplete; recompute all scores
        recompute_scores(search_state)

    return search_state


def save_search_state(search_state):
    """
    Save raw metric results plus cached ranking fields to the JSON state file.

    The saved settings include both weighted-objective weights and Pareto
    distance weights so cached ranking fields can be validated on resume.

    Args:
        search_state (SearchState): The search state object to serialize.
    """
    path = Path(STATE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metric_mode": search_state.mode,
        "metrics": list(search_state.metrics),
        "objective_mode": search_state.objective,
        "weights": dict(search_state.weights),
        "pareto_weights": dict(search_state.pareto_weights),
        "best_epoch": search_state.best_epoch,
        "best_score": search_state.best_score,
        "best_metrics": (
            dict(search_state.best_metrics)
            if search_state.best_metrics is not None
            else None
        ),
        "evaluated_epochs": {},
    }

    for epoch in sorted(search_state.results):
        row = search_state.results[epoch]

        epoch_data = {}

        # Store raw metric values
        for name in search_state.metrics:
            epoch_data[name] = float(row["metrics"][name])

        # Store derived fields (score, pareto)
        epoch_data["score"] = (
            float(row["score"])
            if math.isfinite(row["score"])
            else None
        )
        epoch_data["pareto"] = int(row["pareto"])

        data["evaluated_epochs"][str(epoch)] = epoch_data

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run_metric_script_json(search_state):
    """
    Run the metric script and return its parsed overall metrics.

    Args:
        search_state (SearchState): Search state with metric mode settings.

    Returns:
        dict: Parsed overall metric values.

    Raises:
        RuntimeError: If the metric subprocess fails or the JSON output is invalid.
    """
    tmp_json_path = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix="metric_eval_",
            suffix=".json",
            delete=False,
        ) as handle:
            # Keep the file path after closing the handle so the metric subprocess can
            # write to it on Windows.
            tmp_json_path = handle.name

        log(
            f"Stage 3/3 | Running {search_state.mode.upper()} "
            f"metrics on {SRC_DIR}"
        )

        # Build metric script command
        metric_cmd = [
            sys.executable,
            "-m",
            search_state.metric_module,
            SRC_DIR,
        ]

        # Add full-reference specific arguments if needed
        if search_state.mode == "fr":
            metric_cmd.append(GT_DIR)

            if str(FR_LPIPS_NET).lower() == "vgg":
                metric_cmd.extend(["--lpips-net", "vgg"])

            crop_border = int(FR_CROP_BORDER)
            if crop_border != 0:
                metric_cmd.extend(["--crop-border", str(crop_border)])

            if int(FR_TEST_Y) == 1:
                metric_cmd.append("--test-y")

        # Specify our tmp file as the output file for JSON results
        metric_cmd.extend(["--json-out", tmp_json_path])

        # Run the metric script as a subprocess
        with subprocess.Popen(
            metric_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            # Stream output in real-time
            for line in proc.stdout or []:
                sys.stdout.write(line)
                sys.stdout.flush()

            if proc.wait() != 0:
                raise RuntimeError("Metric script failed")

        with open(tmp_json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        overall = report.get("overall")
        if not isinstance(overall, dict):
            raise RuntimeError("Metric JSON missing overall block")

        # Extract only the metrics we care about
        metrics = {}
        for name in search_state.metrics:
            if name not in overall:
                raise RuntimeError(f"Metric JSON missing overall.{name}")
            metrics[name] = float(overall[name])

        return metrics
    finally:
        # Clean up temporary file
        if tmp_json_path:
            Path(tmp_json_path).unlink(missing_ok=True)


def evaluate_epoch(search_state, epoch):
    """
    Evaluate one checkpoint epoch unless cached metrics already exist.

    The evaluation path extracts generator weights, runs inference, computes
    metrics, updates in-memory ranking fields, and saves the search state.

    Args:
        search_state (SearchState): Search state to read and update.
        epoch (int): Epoch number to evaluate.

    Returns:
        int or None: The evaluated epoch, or None when the epoch was cached.

    Raises:
        RuntimeError: If the checkpoint, extraction, inference, source
        directory, or metric stage fails.
    """
    # Check if this epoch was already evaluated (from cache)
    cached = search_state.results.get(epoch)
    if cached is not None:
        log(f"Epoch {epoch} already evaluated. Cached metrics: {format_metrics(search_state.metrics, cached['metrics'], 6)}")
        return None

    # Locate the checkpoint file for this epoch
    checkpoint_path = Path(CKPT_DIR).joinpath(f"checkpoint_epoch_{epoch}_step_0.pth")
    if not checkpoint_path.exists():
        raise RuntimeError(f"checkpoint not found: {checkpoint_path}")

    log(f"Evaluating epoch {epoch}")
    run_start = time.time()
    
    # Stage 1: Extract generator weights into OUT_PTH for the current epoch.
    # EMA (Exponential Moving Average) weights are typically more stable than
    # regular checkpoint weights, so try EMA first and fall back to G.
    extract_start = time.time()
    print_section_banner(f"Epoch {epoch} | Stage 1/3")
    log(f"Stage 1/3 | Extracting EMA from {checkpoint_path.name}")
    
    extract_cmd = [
        sys.executable,
        "-m",
        EXTRACT_MODULE,
        "--ckpt",
        str(checkpoint_path),
        "--which",
        "EMA",
        "--out",
        OUT_PTH,
    ]
    if subprocess.run(extract_cmd).returncode != 0:
        log(
            f"Stage 1/3 | EMA extraction failed for epoch {epoch}; "
            "trying G weights"
        )
        extract_cmd[extract_cmd.index("EMA")] = "G"

        if subprocess.run(extract_cmd).returncode != 0:
            raise RuntimeError(f"extract_submodel failed for epoch {epoch}")
    
    # Stage 2: Run inference with the extracted weights and save outputs to SRC_DIR.
    # This generates the super-resolved frames that will be scored by metrics.
    test_start = time.time()
    print_section_banner(f"Epoch {epoch} | Stage 2/3")
    log(
        "Stage 2/3 | Running inference "
        f"(tile_t={TILE_T}, tile_h={TILE_H}, tile_w={TILE_W}, "
        f"pad_t={PAD_T}, pad_h={PAD_H}, pad_w={PAD_W}, "
        f"precision={str(TEST_PRECISION).lower()}, "
        f"compile_mode={str(TEST_COMPILE_MODE).lower()})"
    )
    
    inference_cmd = [
        sys.executable,
        TEST_SCRIPT,
        "--tile_t",
        str(TILE_T),
        "--tile_h",
        str(TILE_H),
        "--tile_w",
        str(TILE_W),
        "--pad_t",
        str(PAD_T),
        "--pad_h",
        str(PAD_H),
        "--pad_w",
        str(PAD_W),
        "--precision",
        str(TEST_PRECISION).lower(),
        "--compile_mode",
        str(TEST_COMPILE_MODE).lower(),
        CFG,
        OUT_PTH,
    ]
    
    if subprocess.run(inference_cmd).returncode != 0:
        raise RuntimeError(f"{TEST_SCRIPT} failed")
    
    # Stage 3: Run the metric script on SRC_DIR and parse the overall averages.
    metrics_start = time.time()
    print_section_banner(f"Epoch {epoch} | Stage 3/3")
    
    source_dir = Path(SRC_DIR)
    if not source_dir.is_dir():
        raise RuntimeError(f"Source directory not found: {source_dir}")
    if not any(path.is_dir() for path in source_dir.iterdir()):
        raise RuntimeError(
            f"{SRC_DIR} has no sequence subdirectories "
            "before metric evaluation"
        )
    
    metrics = run_metric_script_json(search_state)

    # Update search state with new epoch results
    old_best_epoch = search_state.best_epoch
    search_state.results[int(epoch)] = {
        "metrics": dict(metrics),
        "score": NEG_INF,
        "pareto": 0,
    }
    recompute_scores(search_state)
    save_search_state(search_state)

    done_time = time.time()
    log(
        "Stage 3/3 | Parsed overall averages: "
        f"{format_metrics(search_state.metrics, metrics, 4)}"
    )
    log(
        f"Epoch {epoch} complete in {int(done_time - run_start)}s | "
        f"stage times: extract={int(test_start - extract_start)}s "
        f"test={int(metrics_start - test_start)}s "
        f"metrics={int(done_time - metrics_start)}s | "
        f"score={search_state.results[epoch]['score']:.6f}"
    )

    # Log if a new best checkpoint was found
    if (
        search_state.best_epoch is not None
        and search_state.best_epoch != old_best_epoch
    ):
        print_section_banner("New Best So Far")
        log(
            f"New best so far: epoch={search_state.best_epoch} "
            f"score={search_state.best_score:.6f} "
            f"({format_metrics(search_state.metrics, search_state.best_metrics, 6)})"
        )

    return epoch


def build_next_candidates(search_state, previous_step, new_step, top_epochs):
    """
    Build the next refinement-round candidate list around the best
    epochs so far.

    Args:
        search_state (SearchState): Search state with best-so-far data.
        previous_step (int): Step size from the previous round.
        new_step (int): Step size for the next round.
        top_epochs (list[int]): Best epochs from the previous round.

    Returns:
        list[int]: Sorted candidate epochs for the next round.
    """
    centers = list(top_epochs)

    # Always refine around the best checkpoint found so far.
    if search_state.best_epoch is not None and search_state.best_epoch not in centers:
        centers.append(search_state.best_epoch)

    # If no refinement centers are available, continue from the range midpoint.
    if not centers:
        centers = [(EPOCH_MIN + EPOCH_MAX) // 2]

    seen = set()

    # For each center epoch, we add candidates in the range [center - previous_step, center + previous_step] 
    # at intervals of new_step. This creates a new grid of candidates around each center.
    for center in centers:
        start = max(EPOCH_MIN, center - previous_step)
        stop = min(EPOCH_MAX, center + previous_step)
        seen.update(range(start, stop + 1, new_step))

    # Early rounds also sample a half-step offset grid to cover gaps between
    # the base grid points.
    if int(USE_OFFSET_GRID) == 1 and search_state.current_round <= int(OFFSET_GRID_ROUNDS):
        shift = new_step // 2

        # The set removes duplicate candidates; keep shifted epochs in range.
        if shift > 0:
            seen.update(
                epoch + shift
                for epoch in list(seen)
                if epoch + shift <= EPOCH_MAX
            )

    return sorted(seen)
    

def run_search():
    """
    Run the coarse-to-fine checkpoint search.

    The search resumes cached results, evaluates candidate epochs round by
    round, refines candidates around the best epochs, and logs the final best
    checkpoint summary.
    """
    search_state = initialize_search_state()
    load_search_state(search_state)
    
    print_banner("Checkpoint Search")
    log(f"Search range: epochs {EPOCH_MIN}..{EPOCH_MAX}")
    log(f"Metric mode: {search_state.mode}")
    log(f"Objective mode: {search_state.objective}")
    log(f"Active metrics: {', '.join(search_state.metrics)}")
    log(f"Initial step: {INITIAL_STEP}")
    log(f"TOPK: {TOPK}")
    log(f"Eval source dir: {SRC_DIR}")
    log(f"Inference script: {TEST_SCRIPT}")
    log(f"State file: {STATE_FILE}")
    if search_state.mode == "fr":
        log(f"Ground-truth dir: {GT_DIR}")
    if search_state.best_epoch is not None:
        log(
            f"Resuming with best so far: epoch={search_state.best_epoch} "
            f"score={search_state.best_score:.6f} "
            f"({format_metrics(search_state.metrics, search_state.best_metrics, 6)})"
        )

    # Constructing the initial candidates
    # Start with a coarse grid covering the entire epoch range
    step = int(INITIAL_STEP)
    offset = int(OFFSET_VALUE) or step // 2
    candidates = set(range(int(EPOCH_MIN), int(EPOCH_MAX) + 1, step))

    # Add offset grid if enabled (explores between base grid points)
    if USE_OFFSET_GRID and offset > 0:
        candidates.update(range(int(EPOCH_MIN) + offset, int(EPOCH_MAX) + 1, step))

    candidates = sorted(candidates)
    
    # Run the search rounds, refining candidates around the best epochs found so far
    for round_index in range(1, int(ROUND_MAX) + 1):
        search_state.current_round = round_index
        print_banner(
            f"Round {round_index} | step={step} | "
            f"candidates={len(candidates)}"
        )
        log_round_candidates(candidates)
        
        round_results = []
        top_epochs = []
        
        for candidate_index, epoch in enumerate(candidates):
            if epoch < EPOCH_MIN or epoch > EPOCH_MAX:
                continue
            
            # Display the remaining candidates at the start of each evaluation for better visibility on long rounds.
            log_round_candidates(candidates[candidate_index + 1:])
            
            result = evaluate_epoch(search_state, epoch)
            
            if result is not None:
                round_results.append(result)
            
        if not round_results:
            log("No new evaluations in this round.")
        else:
            # Sort round results by ranking key and take top_k for next round
            round_results.sort(key=lambda epoch: epoch_rank_key(search_state, epoch))
            top_epochs = round_results[:TOPK]
            log_round_summary(search_state, top_epochs)
        
        # If the step is already at the minimum of 1, then we cannot refine further
        if step <= 1:
            break
        
        # Halve the step size for the next round (coarse-to-fine)
        new_step = max(1, step // 2)
        candidates = build_next_candidates(search_state, step, new_step, top_epochs)
        step = new_step
    
    print_banner("Search Complete")
    
    if search_state.best_epoch is not None:
        log("Best checkpoint found:")
        log(f"  epoch={search_state.best_epoch}")
        log(f"  score={search_state.best_score:.6f}")
        log(f"  {format_metrics(search_state.metrics, search_state.best_metrics, 6)}")
        if search_state.objective == "pareto":
            log(
                "  on Pareto frontier: "
                f"{search_state.results[search_state.best_epoch]['pareto']}"
            )
        log("Outputs:")
        log(f"  Search state saved to: {STATE_FILE}")
    else:
        log("No successful evaluations recorded.")   
    

def main():
    """
    Validate settings and run the checkpoint search.

    Any exception is logged and converted into process exit code 1.
    """
    try:
        validate_settings()
        run_search()
    except Exception as e:
        log(f"ERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
