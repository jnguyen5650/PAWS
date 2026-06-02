"""
Extract a single sub-model weights file from a training checkpoint.

This script reads one of your training checkpoints (the dict you save in
checkpoint.py)
and exports exactly one of:
  - Generator (G)           -> checkpoint["model_state_dict"] (or bare
    state_dict fallback)
  - EMA Generator (EMA)     -> checkpoint["ema_state_dict"]
  - Discriminator (D)       -> checkpoint["discriminator_state_dict"]

The output is a standalone .pth containing ONLY the selected model's state_dict
(with common wrapper/DDP prefixes stripped by default).

Purpose:
- Training checkpoints include optimizer/scheduler/scaler/etc state.
- Inference scripts want a clean state_dict .pth.

Examples:
  python -m tools.extract_submodel \
    --ckpt path/to/checkpoint_epoch_50_step_0.pth --which G
  python -m tools.extract_submodel \
    --ckpt path/to/checkpoint_epoch_50_step_0.pth --which EMA
  python -m tools.extract_submodel \
    --ckpt path/to/checkpoint_epoch_50_step_0.pth --which D

Optional:
  --out path/to/output.pth   (otherwise a default path is derived)
  --no_strip                 (do not strip leading "module." prefixes)
"""

import argparse
import os
import torch


def _is_state_dict_like(obj):
    """
    Heuristic check for whether `obj` looks like a PyTorch state_dict.

    Args:
        obj: Any Python object.

    Returns:
        bool: True if `obj` resembles a state_dict, else False.
    """
    if not isinstance(obj, dict) or len(obj) == 0:
        return False
    v = next(iter(obj.values()))
    return isinstance(v, torch.Tensor)


def _strip_leading_module_prefixes(key):
    """
    Strip leading "module." prefixes from a state_dict key.

    It handles:
      - DDP(model): keys like "module.conv.weight"
      - DDP(CheckpointWrapper(model)): keys like "module.module.conv.weight"
      - CheckpointWrapper(model): keys like "module.conv.weight"

    Args:
        key (str): state_dict key.

    Returns:
        str: key with any number of leading "module." stripped.
    """
    while key.startswith("module."):
        key = key[len("module."):]
    return key


def _clean_state_dict(state_dict):
    """
    Return a cleaned copy of a state_dict with wrapper/DDP prefixes removed.

    Args:
        state_dict (dict): Raw state_dict (keys -> tensors)

    Returns:
        dict: Cleaned state_dict where keys have leading "module." stripped.
    """
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[_strip_leading_module_prefixes(k)] = v
    return cleaned


def _infer_default_out_path(ckpt_path, which):
    """
    Derive a default output path next to the checkpoint file.

    Naming:
      <checkpoint_stem>_G.pth
      <checkpoint_stem>_G_EMA.pth
      <checkpoint_stem>_D.pth

    Args:
        ckpt_path (str): Input checkpoint path.
        which (str): One of {"G","EMA","D"}.

    Returns:
        str: Output path.
    """
    base_dir = os.path.dirname(os.path.abspath(ckpt_path))
    base_name = os.path.splitext(os.path.basename(ckpt_path))[0]

    if which == "G":
        suffix = "G"
    elif which == "EMA":
        suffix = "G_EMA"
    elif which == "D":
        suffix = "D"
    else:
        raise ValueError(f"Unknown which='{which}'. Expected G, EMA, or D.")

    return os.path.join(base_dir, f"{base_name}_{suffix}.pth")


def _select_submodel_state(ckpt_obj, which):
    """
    Select and return the raw state_dict for the requested submodel.

    Expected checkpoint structure (your training checkpoints):
      - Generator weights:       ckpt_obj["model_state_dict"]
      - EMA Generator weights:   ckpt_obj["ema_state_dict"]
      - Discriminator weights:   ckpt_obj["discriminator_state_dict"]

    Also supports a common convenience case:
      - If `which == "G"` and the file is a bare state_dict (not a training
        checkpoint dict), it will export that as the Generator weights.

    Args:
        ckpt_obj: The result of torch.load(...). Usually a dict.
        which (str): One of {"G","EMA","D"}.

    Returns:
        (state_dict, source_label)
        - state_dict (dict): raw (unstripped) state_dict for the requested
          model
        - source_label (str): human-readable label describing where it came
          from

    Raises:
        KeyError: if the requested weights are not present.
        TypeError: if ckpt_obj isn't a dict-like object for your workflow.
    """
    if not isinstance(ckpt_obj, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(ckpt_obj)}. Expected a "
            "dict from torch.save(...)."
        )

    if which == "G":
        # Preferred: standard training checkpoint format
        if (
            "model_state_dict" in ckpt_obj
            and _is_state_dict_like(ckpt_obj["model_state_dict"])
        ):
            return ckpt_obj["model_state_dict"], "model_state_dict"
        # Fallback: bare state_dict file
        if _is_state_dict_like(ckpt_obj):
            return ckpt_obj, "bare_state_dict"
        raise KeyError(
            "Could not find Generator weights. Expected 'model_state_dict' "
            "or a bare state_dict file."
        )

    if which == "EMA":
        if (
            "ema_state_dict" in ckpt_obj
            and _is_state_dict_like(ckpt_obj["ema_state_dict"])
        ):
            return ckpt_obj["ema_state_dict"], "ema_state_dict"
        raise KeyError(
            "Could not find EMA weights. Expected 'ema_state_dict' in the "
            "checkpoint."
        )

    if which == "D":
        if (
            "discriminator_state_dict" in ckpt_obj
            and _is_state_dict_like(ckpt_obj["discriminator_state_dict"])
        ):
            return (
                ckpt_obj["discriminator_state_dict"],
                "discriminator_state_dict",
            )
        raise KeyError(
            "Could not find Discriminator weights. Expected "
            "'discriminator_state_dict' in the checkpoint."
        )

    raise ValueError(f"Unknown which='{which}'. Expected G, EMA, or D.")


def main():
    """
    Loads a checkpoint, selects one submodel, optionally strips leading wrapper
    prefixes, and saves a standalone state_dict .pth.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract Generator / EMA Generator / Discriminator weights from a "
            "training checkpoint."
        )
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        type=str,
        help="Path to the input checkpoint (.pth).",
    )
    parser.add_argument(
        "--which",
        required=True,
        choices=["G", "EMA", "D"],
        help=(
            "Which weights to export: G (Generator), EMA (EMA Generator), "
            "D (Discriminator)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help=(
            "Output .pth path. If not provided, defaults next to checkpoint "
            "with a suffix."
        ),
    )
    parser.add_argument(
        "--no_strip",
        action="store_true",
        help='Do not strip leading "module." prefixes from keys.',
    )
    parser.add_argument(
        "--map_location",
        default="cpu",
        type=str,
        help='torch.load map_location (default "cpu").',
    )

    args = parser.parse_args()

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = args.out or _infer_default_out_path(ckpt_path, args.which)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    ckpt_obj = torch.load(ckpt_path, map_location=args.map_location)
    raw_sd, source_label = _select_submodel_state(ckpt_obj, args.which)

    final_sd = raw_sd if args.no_strip else _clean_state_dict(raw_sd)

    torch.save(final_sd, out_path)

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Export selection: {args.which} (from {source_label})")
    print(f"Stripped prefixes: {'no' if args.no_strip else 'yes'}")
    print(f"Saved state_dict: {out_path}")
    print(f"Num keys: {len(final_sd)}")


if __name__ == "__main__":
    main()
