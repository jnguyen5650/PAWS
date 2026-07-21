"""
Handle model loading, runtime checks, and failure diagnostics.
"""

import importlib
import os

import torch

from helpers import clean_state_dict
from helpers import cuda_supports_bf16
from helpers import ensure_repo_root_on_path
from helpers import load_published_model_artifact


def validate_runtime_args(device, precision, compile_mode):
    """
    Validate the requested precision and compile settings.

    Args:
        device (torch.device): Active torch device.
        precision (str): Inference precision mode.
        compile_mode (str): Requested compile mode.

    Returns:
        None: Raises an exception if the settings are invalid.

    Raises:
        ValueError: If the selected precision is unsupported for the active device.
        RuntimeError: If ``torch.compile`` is unavailable for the current runtime.
    """
    if precision != "fp32" and device.type != "cuda":
        raise ValueError(
            f"Precision {precision} requires CUDA, but the active device is {device.type}."
        )

    if precision == "bf16" and not cuda_supports_bf16():
        raise ValueError(
            "Precision bf16 is not supported by the current CUDA GPU. Use fp32 instead."
        )

    if compile_mode != "none" and not hasattr(torch, "compile"):
        raise RuntimeError(
            f"torch.compile is unavailable in this PyTorch runtime: {torch.__version__}."
        )


def maybe_compile_model(model, compile_mode):
    """
    Compile a model with ``torch.compile`` when requested.

    Args:
        model (torch.nn.Module): Loaded generator model.
        compile_mode (str): Requested compile mode.

    Returns:
        tuple: Compiled model and optional compile error.
    """
    if compile_mode == "none":
        return model, None

    try:
        return torch.compile(model, dynamic=True, mode=compile_mode), None
    except Exception as exc:
        return None, exc


def make_model_runtime(
    eager_model, model_settings, progress_cb=None, total_frames=0, processed=0
):
    """
    Build the runtime state for eager or compiled inference.

    Args:
        eager_model (torch.nn.Module): Loaded eager model.
        model_settings (dict): Inference settings dictionary.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        processed (int): Already processed frame count.

    Returns:
        dict: Runtime dictionary for eager and compiled execution.
    """
    compile_mode = model_settings.get("compile_mode", "none")
    # Keep both the eager and currently active model reference so compile mode
    # can be disabled later without reloading weights from disk.
    runtime = {
        "eager_model": eager_model,
        "active_model": eager_model,
        "using_compile": False,
        "compile_mode": compile_mode,
        "compile_first_forward_logged": False,
    }
    if compile_mode == "none":
        return runtime

    message = (
        f"torch.compile enabled with mode '{compile_mode}'. "
        "The first compiled forward may take several minutes."
    )
    log_runtime_message("compile", message)
    notify_runtime_status(progress_cb, total_frames, message, processed)

    compiled_model, compile_error = maybe_compile_model(eager_model, compile_mode)
    if compiled_model is not None:
        runtime["active_model"] = compiled_model
        runtime["using_compile"] = True
        return runtime

    message = (
        "torch.compile setup failed; continuing with eager inference. "
        f"Reason: {compact_exception_message(compile_error)}"
    )
    log_runtime_message("compile", message)
    notify_runtime_status(progress_cb, total_frames, message, processed)
    return runtime


def get_active_model(model_runtime):
    """
    Return the currently active model object.

    Args:
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.

    Returns:
        object: Active torch model.
    """
    if isinstance(model_runtime, dict):
        active_model = model_runtime.get("active_model")
        if active_model is not None:
            return active_model
        return model_runtime.get("eager_model")
    return model_runtime


def disable_compiled_model(model_runtime):
    """
    Switch a compiled runtime back to eager inference.

    Args:
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.

    Returns:
        bool: True when compile mode was disabled.
    """
    if not isinstance(model_runtime, dict):
        return False
    if not model_runtime.get("using_compile"):
        return False
    model_runtime["active_model"] = model_runtime.get("eager_model")
    model_runtime["using_compile"] = False
    return True


def maybe_log_first_compiled_forward(
    model_runtime, progress_cb, total_frames, processed
):
    """
    Log the first compiled forward pass.

    Args:
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        processed (int): Already processed frame count.

    Returns:
        None: Logs and notifies the UI about the first compiled forward.
    """
    if not isinstance(model_runtime, dict):
        return
    if not model_runtime.get("using_compile"):
        return
    if model_runtime.get("compile_first_forward_logged"):
        return

    # Emit the long compile warning once so the UI does not repeat it for every
    # window while TorchInductor is warming up.
    model_runtime["compile_first_forward_logged"] = True
    message = "Starting first compiled forward; TorchInductor compilation may take several minutes."
    log_runtime_message("compile", message)
    notify_runtime_status(progress_cb, total_frames, message, processed)


def get_model_class(architecture):
    """
    Load a model class from the repo registry.

    Args:
        architecture (str): Model registry key.

    Returns:
        object: Registered model class.
    """
    ensure_repo_root_on_path()
    importlib.import_module("models")
    registry = importlib.import_module("models.registry")
    return registry.get_model_cls(architecture)


def build_model_kwargs(artifact, model_settings):
    """
    Build model constructor kwargs from artifact metadata.

    Args:
        artifact (dict): Published model artifact dictionary.
        model_settings (dict): Inference settings dictionary.

    Returns:
        dict: Constructor kwargs for the model.
    """
    model_kwargs = dict(artifact.get("model_kwargs") or {})
    for key in ("name", "flow_param_keywords"):
        model_kwargs.pop(key, None)

    if (
        "is_sequential_cleaning" in model_kwargs
        and "is_sequential_cleaning" in model_settings
    ):
        model_kwargs["is_sequential_cleaning"] = bool(
            model_settings.get("is_sequential_cleaning")
        )

    return model_kwargs


def load_inference_model(model_path, model_settings, device):
    """
    Load and prepare the published model for inference.

    Args:
        model_path (str): Published model path.
        model_settings (dict): Inference settings dictionary.
        device (torch.device): Active torch device.

    Returns:
        tuple: Loaded model and published artifact dictionary.

    Raises:
        RuntimeError: If the artifact cannot be loaded or is structurally invalid.
    """
    artifact, err = load_published_model_artifact(model_path, device)
    if err:
        raise RuntimeError(err)

    architecture = artifact.get("architecture")
    model_cls = get_model_class(architecture)
    model_kwargs = build_model_kwargs(artifact, model_settings)
    state_dict = artifact.get("model_state_dict")

    if not isinstance(state_dict, dict):
        raise RuntimeError("Published model has an invalid model_state_dict.")

    model = model_cls(**model_kwargs).to(device)
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.eval()
    return model, artifact


def run_forward(model, lr_clip, precision):
    """
    Run one model forward pass.

    Args:
        model (torch.nn.Module): Loaded generator model.
        lr_clip (torch.Tensor): Input clip on the active device.
        precision (str): Inference precision mode.

    Returns:
        torch.Tensor: Super-resolved clip tensor on the active device.
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

    sr_clip = pred[0] if isinstance(pred, (tuple, list)) else pred
    return sr_clip.clamp(0, 1).float().squeeze(0)


def log_runtime_message(channel, message):
    """
    Print a tagged runtime message.

    Args:
        channel (str): Runtime channel name.
        message (str): Message to print.

    Returns:
        None: Prints the message to stdout with a channel tag.
    """
    print(f"[PAWS {channel}] {message}", flush=True)


def compact_exception_message(exc):
    """
    Collapse an exception into one short line.

    Args:
        exc (Exception): Exception object.

    Returns:
        str: Compact message string.
    """
    message = str(exc).strip().replace("\r", " ").replace("\n", " ")
    if len(message) > 240:
        return message[:237] + "..."
    return message


def format_detail_value(value):
    """
    Normalize a value for failure diagnostics.

    Args:
        value (object): Value to stringify.

    Returns:
        str: Printable value or ``n/a``.
    """
    if value in (None, ""):
        return "n/a"
    return str(value)


def build_failure_technical_details(
    input_path,
    output_path,
    model_settings,
    out,
    failure_summary,
    exc=None,
    traceback_text="",
):
    """
    Build multi-line diagnostics for a failed item.

    Args:
        input_path (str): Source video path.
        output_path (str): Output file or folder path.
        model_settings (dict): Inference settings dictionary.
        out (dict): Output settings dictionary.
        failure_summary (str): Short visible failure summary.
        exc (Exception | None): Optional exception object.
        traceback_text (str): Optional traceback text.

    Returns:
        str: Multi-line debugging text.
    """

    def _safe_name(path):
        if not path:
            return "n/a"
        try:
            name = os.path.basename(os.path.normpath(str(path)))
        except Exception:
            name = ""
        return name or "n/a"

    metadata = model_settings.get("metadata") or {}
    architecture = metadata.get("architecture") or model_settings.get("architecture")
    scale_factor = metadata.get("scale_factor") or model_settings.get("scale_factor")
    scale_text = format_detail_value(scale_factor)
    if scale_text != "n/a":
        scale_text = f"x{scale_text}"

    model_display = (metadata.get("display_name") or "").strip()
    if not model_display:
        model_display = _safe_name(model_settings.get("model_path"))

    def _tile_val(resolved_key, requested_key):
        if resolved_key in model_settings:
            return format_detail_value(model_settings.get(resolved_key))
        return format_detail_value(model_settings.get(requested_key))

    lines = [
        f"Failure Summary: {failure_summary}",
        f"Exception Type: {exc.__class__.__name__ if exc is not None else 'n/a'}",
        "",
        "Input",
        f"  Source File: {_safe_name(input_path)}",
        f"  Output Target: {_safe_name(output_path)}",
        "",
        "Model",
        f"  Display Name: {model_display}",
        f"  Architecture: {format_detail_value(architecture)}",
        f"  Scale Factor: {scale_text}",
        "",
        "Runtime",
        f"  Precision: {format_detail_value(model_settings.get('precision'))}",
        f"  Compile Mode: {format_detail_value(model_settings.get('compile_mode'))}",
        f"  Temporal Tile: {_tile_val('resolved_tile_t', 'tile_t')}",
        f"  Spatial Tile Height: {_tile_val('resolved_tile_h', 'tile_h')}",
        f"  Spatial Tile Width: {_tile_val('resolved_tile_w', 'tile_w')}",
        f"  Temporal Padding: {format_detail_value(model_settings.get('pad_t'))}",
        f"  Spatial Padding Height: {format_detail_value(model_settings.get('pad_h'))}",
        f"  Spatial Padding Width: {format_detail_value(model_settings.get('pad_w'))}",
        "",
        "Output",
        f"  Export Mode: {format_detail_value(out.get('export_mode'))}",
        f"  Container: {format_detail_value(out.get('container_format'))}",
        f"  Video Codec: {format_detail_value(out.get('video_codec'))}",
        f"  Encoding Mode: {format_detail_value(out.get('encoding_mode'))}",
        f"  CRF Value: {format_detail_value(out.get('crf_value'))}",
        f"  Bitrate (kbps): {format_detail_value(out.get('bitrate_kbps'))}",
        f"  Preset: {format_detail_value(out.get('preset'))}",
    ]

    traceback_text = (traceback_text or "").rstrip()
    if traceback_text:
        lines.extend(["", "Traceback", traceback_text])

    return "\n".join(lines)


def make_exception_search_text(exc):
    """
    Build searchable text from an exception chain.

    Args:
        exc (Exception): Exception object.

    Returns:
        str: Lowercase text built from the exception chain.
    """
    parts = []
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(current.__class__.__name__)
        parts.append(current.__class__.__module__)
        parts.append(str(current))
        current = current.__cause__ or current.__context__
    return " ".join(parts).lower()


def is_torch_compile_failure(exc):
    """
    Detect torch.compile-specific failures.

    Args:
        exc (Exception): Exception raised during inference.

    Returns:
        bool: True when the exception looks compile-specific.
    """
    text = make_exception_search_text(exc)
    # TorchInductor failures percolate up through several wrapper exceptions, so
    # this stays heuristic and scans the full exception chain text.
    markers = (
        "cppcompileerror",
        "backendcompilerfailed",
        "compiler: cl is not found",
        "c++ compile error",
        "failed to compile",
        "torch._inductor",
        "torchinductor",
    )
    return any(marker in text for marker in markers)


def notify_runtime_status(progress_cb, total_frames, message, processed=0):
    """
    Send a runtime status message to the UI callback.

    Args:
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        message (str): Message to display.
        processed (int): Number of processed frames.

    Returns:
        None: Sends the status message to the progress callback.
    """
    if progress_cb is None or not message:
        return
    progress_cb(processed, total_frames, message)


def is_cuda_out_of_memory(exc):
    """
    Detect CUDA out-of-memory failures.

    Args:
        exc (Exception): Runtime exception raised by torch.

    Returns:
        bool: True when the error looks like CUDA OOM.
    """
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cudnn_status_alloc_failed" in message
        or "cublas_status_alloc_failed" in message
    )


def clear_cuda_runtime_cache(device):
    """
    Clear CUDA allocator state after a failure.

    Args:
        device (torch.device): Active torch device.

    Returns:
        None: Synchronizes and clears the CUDA cache.
    """
    if device.type != "cuda":
        return
    try:
        with torch.cuda.device(device):
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
    except Exception:
        pass
