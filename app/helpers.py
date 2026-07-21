"""
Collect filesystem, metadata, upload, and published-model helpers.
"""

import os
import sys
import tempfile
from pathlib import Path

import av
import torch

from constants import PAWS_MODEL_FORMAT
from constants import PAWS_MODEL_VERSION
from constants import PUBLISHED_MODEL_EXT
from constants import VIDEO_EXTS


def quiet_close(resource):
    """
    Close a resource while ignoring cleanup errors.

    Args:
        resource (object): Object with a close method, or None.

    Returns:
        None: Closes the resource if it is not None.
    """
    try:
        if resource is not None:
            resource.close()
    except Exception:
        pass


def cuda_supports_bf16():
    """
    Return whether the active CUDA runtime supports bf16.

    Returns:
        bool: True when CUDA bf16 is available.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def quiet_remove_file(path):
    """
    Delete a file while ignoring cleanup errors.

    Args:
        path (str): File path to delete.

    Returns:
        None: Deletes the file if it exists.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def first_stream(container, stream_type):
    """
    Return the first stream with the requested type.

    Args:
        container (av.container.input.InputContainer): Open PyAV container.
        stream_type (str): Stream type like ``video`` or ``audio``.

    Returns:
        object: Matching stream, or None when missing.
    """
    for stream in container.streams:
        if stream.type == stream_type:
            return stream
    return None


def safe_mkdir(path):
    """
    Create a directory and return success status.

    Args:
        path (str): Directory path to create.

    Returns:
        tuple: Success flag and optional error string.
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True, None
    except Exception as exc:
        return False, str(exc)


def ensure_output_dir(out):
    """
    Make sure the configured output directory is ready.

    Args:
        out (dict): Output settings dictionary.

    Returns:
        str | None: Error text, or None when the directory is ready.
    """
    if os.path.exists(out["output_dir"]):
        return None
    ok, err = safe_mkdir(out["output_dir"])
    if ok:
        return None
    return f"Could not create output directory: {err}"


def parse_codec_name(codec_ui):
    """
    Map a UI codec label to an ffmpeg codec name.

    Args:
        codec_ui (str): Codec label from the UI.

    Returns:
        str: Codec name, or an empty string.
    """
    if not codec_ui:
        return ""
    for prefix in ("libx264", "libx265", "libvpx-vp9"):
        if codec_ui.startswith(prefix):
            return prefix
    return ""


def get_video_info(file_path):
    """
    Read basic metadata from a video file.

    Args:
        file_path (str): Video path to inspect.

    Returns:
        tuple: Metadata dictionary and optional error string.
    """
    container = None
    try:
        container = av.open(file_path)
        video_stream = first_stream(container, "video")
        if video_stream is None:
            return None, "No video stream found."

        duration = 0.0
        if video_stream.duration is not None and video_stream.time_base is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        elif container.duration:
            duration = float(container.duration / 1_000_000)

        return {
            "width": int(video_stream.width or 0),
            "height": int(video_stream.height or 0),
            "duration_sec": duration,
            "fps": (
                float(video_stream.average_rate) if video_stream.average_rate else 30.0
            ),
            "codec": video_stream.codec.name if video_stream.codec else "unknown",
        }, None
    except Exception as exc:
        return None, str(exc)
    finally:
        quiet_close(container)


def crf_quality_assessment(codec, crf):
    """
    Map a CRF value to a simple quality label.

    Args:
        codec (str): Codec name.
        crf (int): CRF integer value.

    Returns:
        str: Quality label for the current range.
    """
    codec = (codec or "").lower()
    crf = int(crf)

    if codec in ("libx264", "libx265"):
        if crf == 0:
            return "Lossless"
        if crf <= 17:
            return "Very high"
        if crf <= 23:
            return "High"
        if crf <= 28:
            return "Good"
        if crf <= 35:
            return "Low"
        return "Very low"

    if codec == "libvpx-vp9":
        if crf == 0:
            return "Lossless"
        if crf <= 15:
            return "Very high"
        if crf <= 30:
            return "High"
        if crf <= 40:
            return "Good"
        if crf <= 50:
            return "Low"
        return "Very low"

    return "N/A"


def estimate_total_frames(container_in, video_stream):
    """
    Estimate the frame count for progress reporting.

    Args:
        container_in (av.container.input.InputContainer): Open input container.
        video_stream (object): Input video stream.

    Returns:
        int: Estimated total frame count.
    """
    if video_stream.frames:
        return int(video_stream.frames)
    fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
    duration = (
        float(container_in.duration / 1_000_000) if container_in.duration else 0.0
    )
    if duration > 0:
        return int(duration * fps)
    return 0


def make_result(ok, message, output_path, is_png, technical_details=None):
    """
    Build the standard per-file result payload.

    Args:
        ok (bool): Success flag.
        message (str): Human-readable result message.
        output_path (str): Output file or folder path.
        is_png (bool): True when the output is a PNG sequence.
        technical_details (str | None): Optional debugging text.

    Returns:
        dict: Standard result dictionary.
    """
    result = {
        "ok": bool(ok),
        "message": str(message),
        "output_path": str(output_path),
        "is_png": bool(is_png),
    }
    if technical_details:
        result["technical_details"] = str(technical_details)
    return result


def list_videos_in_folder(folder):
    """
    List supported video files in one folder.

    Args:
        folder (str): Folder path from the UI.

    Returns:
        tuple: Sorted file paths and optional error string.
    """
    if not folder:
        return [], None

    folder_path = Path(folder)
    if not folder_path.exists():
        return [], "Folder does not exist yet. Type the full path or create it first."
    if not folder_path.is_dir():
        return [], "Path is not a directory."

    try:
        candidates = [
            str(path)
            for path in folder_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTS
        ]
        return sorted(candidates), None
    except PermissionError:
        return [], "Permission denied. Please select a folder you have access to."
    except Exception as exc:
        return [], str(exc)


def write_upload_to_temp(uploaded_file):
    """
    Write an uploaded file into a temporary path.

    Args:
        uploaded_file (object): Streamlit uploaded file object.

    Returns:
        tuple: Temp file path and optional error string.
    """
    try:
        ext = os.path.splitext(uploaded_file.name)[1]
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(uploaded_file.read())
        return tmp_file.name, None
    except Exception as exc:
        return None, str(exc)


def ensure_repo_root_on_path():
    """
    Add the repository root to ``sys.path`` when needed.

    Returns:
        str: Repository root path string.
    """
    repo_root = Path(__file__).resolve().parent.parent
    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    return repo_root_text


def torch_load_file(path, map_location):
    """
    Load a PyTorch file across PyTorch versions.

    Args:
        path (str): File path to load.
        map_location (str | torch.device): Location passed into ``torch.load``.

    Returns:
        object: Loaded PyTorch object.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_leading_module_prefixes(key):
    """
    Strip leading ``module.`` prefixes from a state-dict key.

    Args:
        key (str): State-dict key string.

    Returns:
        str: Key string without leading module prefixes.
    """
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def clean_state_dict(state_dict):
    """
    Build a copy of a state dict with wrapper prefixes removed.

    Args:
        state_dict (dict): Raw state dict.

    Returns:
        dict: Cleaned state dict.
    """
    return {
        strip_leading_module_prefixes(key): value for key, value in state_dict.items()
    }


def validate_published_artifact(artifact, path):
    """
    Check the basic structure of a published PAWS artifact.

    Args:
        artifact (object): Loaded artifact object.
        path (str): Artifact path for error messages.

    Returns:
        str | None: Error text, or None when the artifact is valid.
    """
    if not isinstance(artifact, dict):
        return f"Published model is not a dictionary: {path}"
    if artifact.get("paws_model_format") != PAWS_MODEL_FORMAT:
        return f"Not a PAWS published model: {path}"
    if int(artifact.get("format_version", 0) or 0) != PAWS_MODEL_VERSION:
        return f"Unsupported PAWS model format version in: {path}"
    if "model_state_dict" not in artifact:
        return f"Published model is missing model_state_dict: {path}"
    if "architecture" not in artifact:
        return f"Published model is missing architecture: {path}"
    if "model_kwargs" not in artifact:
        return f"Published model is missing model_kwargs: {path}"
    return None


def load_published_model_artifact(path, map_location):
    """
    Load and validate a published PAWS artifact.

    Args:
        path (str): Artifact path.
        map_location (str | torch.device): Location passed into ``torch.load``.

    Returns:
        tuple: Loaded artifact dictionary and optional error string.
    """
    if not path or not os.path.exists(path):
        return None, f"Published model not found: {path}"

    try:
        artifact = torch_load_file(path, map_location)
    except Exception as exc:
        return None, str(exc)

    err = validate_published_artifact(artifact, path)
    if err:
        return None, err
    return artifact, None


def metadata_from_artifact(artifact, path):
    """
    Extract lightweight UI metadata from a published artifact.

    Args:
        artifact (dict): Published artifact dictionary.
        path (str): Artifact path.

    Returns:
        dict: Metadata dictionary for UI display and inference setup.
    """
    source = artifact.get("source") or {}
    return {
        "path": str(path),
        "display_name": artifact.get("display_name") or Path(path).stem,
        "architecture": artifact.get("architecture") or "unknown",
        "scale_factor": int(artifact.get("scale_factor", 1) or 1),
        "weight_source": artifact.get("weight_source") or "unknown",
        "source_config": source.get("config_path", ""),
        "source_checkpoint": source.get("checkpoint_path", ""),
        "source_state_key": source.get("state_key", ""),
        "model_kwargs": dict(artifact.get("model_kwargs") or {}),
    }


def load_published_model_metadata(path):
    """
    Load display metadata from a published model artifact.

    Args:
        path (str): Artifact path.

    Returns:
        tuple: Metadata dictionary and optional error string.
    """
    artifact, err = load_published_model_artifact(path, "cpu")
    if err:
        return None, err
    return metadata_from_artifact(artifact, path), None


def list_published_models(model_dir):
    """
    List published PAWS artifacts in a directory.

    Args:
        model_dir (str): Model directory path.

    Returns:
        tuple: Sorted artifact paths and optional error string.
    """
    if not model_dir:
        return [], None

    folder = Path(model_dir)
    if not folder.exists():
        return [], "Model directory does not exist yet."
    if not folder.is_dir():
        return [], "Model path is not a directory."

    try:
        paths = [
            str(path)
            for path in folder.iterdir()
            if path.is_file() and path.name.endswith(PUBLISHED_MODEL_EXT)
        ]
        return sorted(paths), None
    except PermissionError:
        return [], "Permission denied. Choose a model directory you can read."
    except Exception as exc:
        return [], str(exc)
