"""
Publish a portable PAWS model artifact from a training checkpoint.

This script loads a training config plus a checkpoint or bare state_dict,
selects generator or EMA weights, strips wrapper prefixes when requested,
and writes a `.paws.pth` artifact with embedded model metadata.

Purpose:
- Package weights and model construction metadata into one portable file.
- Produce an artifact that demo or inference tools can load directly.

Examples:
  python -m tools.publish_model --config configs/model.yaml \
    --ckpt checkpoint_epoch_50_step_0.pth
  python -m tools.publish_model --config configs/model.yaml \
    --ckpt model_G_EMA.pth --which EMA
"""

from utils.traceback import *  # noqa: F401, F403
import argparse
import os
import re
import time
from pathlib import Path

import torch

from utils.config import load_config


PAWS_MODEL_FORMAT = "paws_published_model"
PAWS_MODEL_VERSION = 1
PUBLISHED_MODEL_EXT = ".paws.pth"
DEFAULT_OUTPUT_DIR = os.path.join("outputs", "published_models")


def torch_load_file(path, map_location):
    """
    Load a PyTorch file across PyTorch versions with one stable helper.

    Args:
        path (str): File path to load.
        map_location (str): Device or location string passed into torch.load.

    Returns:
        object: Loaded PyTorch object.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def is_state_dict_like(obj):
    """
    Check whether an object resembles a PyTorch state_dict.

    Args:
        obj (object): Object to inspect.

    Returns:
        bool: True when the object looks like a state_dict.
    """
    if not isinstance(obj, dict) or not obj:
        return False
    value = next(iter(obj.values()))
    return isinstance(value, torch.Tensor)


def strip_leading_module_prefixes(key):
    """
    Remove leading "module." prefixes from a state_dict key.

    Args:
        key (str): State_dict key string.

    Returns:
        str: Key string without leading module prefixes.
    """
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def clean_state_dict(state_dict):
    """
    Build a copy of a state_dict without wrapper prefixes.

    Args:
        state_dict (dict): Raw state_dict.

    Returns:
        dict: Cleaned state_dict.
    """
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[strip_leading_module_prefixes(key)] = value
    return cleaned


def infer_bare_state_source_from_name(checkpoint_path):
    """
    Infer the weight source label for a bare state_dict file from its filename.

    Args:
        checkpoint_path (str): Checkpoint path used for filename inspection.

    Returns:
        str or None: Weight source label, or None when the filename is
        ambiguous.

    Raises:
        ValueError: If the filename appears to describe discriminator weights.
    """
    stem = Path(checkpoint_path).stem.lower()
    tokens = [token for token in re.split(r"[^a-z0-9]+", stem) if token]

    if tokens and tokens[-1] in ("d", "disc", "discriminator"):
        raise ValueError(
            "Checkpoint filename looks like discriminator weights, which "
            "cannot be published for demo inference."
        )

    if "ema" in tokens:
        return "EMA"

    if tokens and tokens[-1] == "g":
        return "G"

    return None


def resolve_bare_state_source(checkpoint_path, which):
    """
    Resolve the weight source label for a bare state_dict checkpoint.

    Args:
        checkpoint_path (str): Checkpoint path used for filename inspection.
        which (str): Requested weight source label.

    Returns:
        str: Resolved weight source label.

    Raises:
        KeyError: If the bare state_dict source cannot be determined.
    """
    inferred = infer_bare_state_source_from_name(checkpoint_path)
    if which in ("G", "EMA"):
        return which

    if inferred:
        return inferred

    raise KeyError(
        "Bare state_dict source is ambiguous. Rename the file with a clear "
        "_G.pth or _G_EMA.pth suffix, "
        "or pass --which G or --which EMA."
    )


def select_submodel_state(checkpoint, which, checkpoint_path):
    """
    Select generator or EMA weights from a checkpoint object.

    Args:
        checkpoint (dict): Loaded checkpoint object.
        which (str): Weight source label: "auto", "G", or "EMA".
        checkpoint_path (str): Checkpoint path used for bare state_dict
            filename inference.

    Returns:
        tuple: `(state_dict, source_key, resolved_which)`, where
        `state_dict` is the selected weights dict, `source_key` describes
        where it came from, and `resolved_which` is the final weight source
        label.

    Raises:
        TypeError: If checkpoint is not a dict-like object for this workflow.
        KeyError: If matching generator or EMA weights cannot be found.
        ValueError: If the requested weight selection is invalid.
    """
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if which == "auto":
        if (
            "ema_state_dict" in checkpoint
            and is_state_dict_like(checkpoint["ema_state_dict"])
        ):
            return checkpoint["ema_state_dict"], "ema_state_dict", "EMA"
        if (
            "model_state_dict" in checkpoint
            and is_state_dict_like(checkpoint["model_state_dict"])
        ):
            return checkpoint["model_state_dict"], "model_state_dict", "G"
        if is_state_dict_like(checkpoint):
            resolved = resolve_bare_state_source(checkpoint_path, which)
            return checkpoint, f"bare_state_dict_{resolved}", resolved
        raise KeyError(
            "Could not find generator or EMA weights in the checkpoint."
        )

    if which == "G":
        if (
            "model_state_dict" in checkpoint
            and is_state_dict_like(checkpoint["model_state_dict"])
        ):
            return checkpoint["model_state_dict"], "model_state_dict", "G"
        if is_state_dict_like(checkpoint):
            resolved = resolve_bare_state_source(checkpoint_path, which)
            return checkpoint, f"bare_state_dict_{resolved}", resolved
        raise KeyError(
            "Could not find generator weights in model_state_dict or a bare "
            "state dict."
        )

    if which == "EMA":
        if (
            "ema_state_dict" in checkpoint
            and is_state_dict_like(checkpoint["ema_state_dict"])
        ):
            return checkpoint["ema_state_dict"], "ema_state_dict", "EMA"
        if is_state_dict_like(checkpoint):
            resolved = resolve_bare_state_source(checkpoint_path, which)
            return checkpoint, f"bare_state_dict_{resolved}", resolved
        raise KeyError("Could not find EMA weights in ema_state_dict.")

    raise ValueError(f"Unknown weight selection: {which}")


def sanitize_model_config(config):
    """
    Extract only the model construction settings from a training config.

    Args:
        config (dict): Loaded YAML config dictionary.

    Returns:
        tuple: `(architecture, model_kwargs)` where `architecture` is the model
        name and `model_kwargs` contains the remaining constructor kwargs.

    Raises:
        KeyError: If the config is missing `model.name`.
    """
    model_config = dict(config.get("model") or {})
    architecture = model_config.get("name")
    if not architecture:
        raise KeyError("Config is missing model.name.")

    model_config.pop("name", None)
    model_config.pop("flow_param_keywords", None)
    return architecture, model_config


def make_safe_name(name):
    """
    Convert a display name into a filesystem-friendly filename stem.

    Args:
        name (str): Proposed display name.

    Returns:
        str: Safe filename stem.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("._")
    if not cleaned:
        return "paws_model"
    return cleaned


def infer_default_output_path(config, which):
    """
    Build the default output path for a published model artifact.

    Args:
        config (dict): Loaded YAML config dictionary.
        which (str): Weight source label.

    Returns:
        str: Output path string.
    """
    logging_config = config.get("logging") or {}
    experiment_name = (
        logging_config.get("name")
        or config.get("name")
        or "paws_model"
    )
    suffix = "G_EMA" if which == "EMA" else "G"
    filename = (
        f"{make_safe_name(experiment_name)}_{suffix}{PUBLISHED_MODEL_EXT}"
    )
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def build_artifact(
    config,
    checkpoint_path,
    config_path,
    state_dict,
    source_key,
    which,
    display_name,
):
    """
    Build the published PAWS model artifact dictionary.

    Args:
        config (dict): Loaded YAML config dictionary.
        checkpoint_path (str): Source checkpoint path.
        config_path (str): Source config path.
        state_dict (dict): Selected model state_dict.
        source_key (str): Source state key label.
        which (str): Weight source label.
        display_name (str or None): Optional display name override.

    Returns:
        dict: Published PAWS model artifact dictionary.
    """
    architecture, model_kwargs = sanitize_model_config(config)
    scale_factor = int(model_kwargs.get("scale_factor", 1) or 1)
    logging_config = config.get("logging") or {}
    chosen_display_name = (
        display_name
        or logging_config.get("name")
        or Path(checkpoint_path).stem
    )

    return {
        "paws_model_format": PAWS_MODEL_FORMAT,
        "format_version": PAWS_MODEL_VERSION,
        "display_name": chosen_display_name,
        "architecture": architecture,
        "scale_factor": scale_factor,
        "weight_source": which,
        "model_kwargs": model_kwargs,
        "model_state_dict": state_dict,
        "source": {
            "config_path": Path(config_path).name,
            "checkpoint_path": Path(checkpoint_path).name,
            "state_key": source_key,
            "published_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }


def parse_args():
    """
    Parse the command-line arguments for model publishing.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Publish a PAWS generator checkpoint with embedded model metadata."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Training config YAML used to build the model.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint or bare state_dict .pth file.",
    )
    parser.add_argument(
        "--which",
        choices=["auto", "G", "EMA"],
        default="auto",
        help=(
            "Weights to publish. auto prefers EMA in full checkpoints and "
            "infers bare state dict files by name."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output .paws.pth path. If omitted, writes under "
            "outputs/published_models using "
            "<experiment>_<G|G_EMA>.paws.pth."
        ),
    )
    parser.add_argument(
        "--display_name",
        default=None,
        help=(
            "Display name stored in the published artifact for demo UI; does "
            "not change the output filename. If omitted, uses config "
            "logging.name, then the checkpoint filename."
        ),
    )
    parser.add_argument(
        "--no_strip",
        action="store_true",
        help="Do not strip leading module prefixes.",
    )
    parser.add_argument(
        "--map_location",
        default="cpu",
        help="torch.load map_location value.",
    )
    return parser.parse_args()


def main():
    """
    Publish a portable PAWS model artifact.

    Returns:
        None: Writes the published artifact to disk and prints a short summary.
    """
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    config = load_config(args.config)
    checkpoint = torch_load_file(args.ckpt, args.map_location)
    raw_state, source_key, resolved_which = select_submodel_state(
        checkpoint,
        args.which,
        args.ckpt,
    )
    final_state = raw_state if args.no_strip else clean_state_dict(raw_state)
    artifact = build_artifact(
        config,
        args.ckpt,
        args.config,
        final_state,
        source_key,
        resolved_which,
        args.display_name,
    )

    output_path = args.out or infer_default_output_path(config, resolved_which)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    torch.save(artifact, output_path)

    print(f"Published model: {output_path}")
    print(f"Architecture: {artifact['architecture']}")
    print(f"Scale factor: {artifact['scale_factor']}")
    print(f"Weight source: {artifact['weight_source']} ({source_key})")
    print(f"State dict keys: {len(final_state)}")


if __name__ == "__main__":
    main()
