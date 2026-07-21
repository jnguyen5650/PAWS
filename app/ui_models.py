"""
Render published-model, tiling, and runtime controls.
"""

import html
import os
from pathlib import Path

import streamlit as st

from constants import COMPILE_MODE_OPTIONS
from constants import DEFAULT_MODEL_DIR
from constants import DEFAULT_PAD_H
from constants import DEFAULT_PAD_W
from constants import DEFAULT_TEMPORAL_CHUNK
from constants import DEFAULT_TILE_HEIGHT
from constants import DEFAULT_TILE_WIDTH
from constants import ICON_CREATE
from constants import ICON_INFO
from constants import ICON_SPARKLES
from constants import ICON_WARNING
from constants import MIN_SPATIAL_TILE
from constants import MIN_TEMPORAL_TILE
from constants import PRECISION_OPTIONS
from helpers import cuda_supports_bf16
from helpers import list_published_models
from helpers import load_published_model_metadata
from helpers import safe_mkdir


@st.cache_data(show_spinner=False)
def cached_model_metadata(model_path, cache_mtime, cache_size):
    """
    Cache published-model metadata for the UI.

    Args:
        model_path (str): Published model path.
        cache_mtime (float): File modification time included in the Streamlit cache key.
        cache_size (int): File size included in the Streamlit cache key.

    Returns:
        tuple: Metadata dictionary and optional error string.
    """
    return load_published_model_metadata(model_path)


def read_model_metadata_for_ui(model_path):
    """
    Load published-model metadata with cache keys.

    Args:
        model_path (str): Published model path.

    Returns:
        tuple: Metadata dictionary and optional error string.
    """
    try:
        stat = os.stat(model_path)
    except Exception as exc:
        return None, str(exc)
    # Include file metadata in the cache key so Streamlit invalidates cached
    # model metadata when the published artifact changes on disk.
    return cached_model_metadata(model_path, stat.st_mtime, stat.st_size)


def render_model_directory_controls():
    """
    Render the published-model directory controls.

    Returns:
        list: Current published-model paths.
    """
    st.header(f"{ICON_SPARKLES} Model Inference")
    model_dir_input = st.text_input(
        "Published Model Directory",
        value=st.session_state.get("model_dir", DEFAULT_MODEL_DIR),
        help=(
            "Folder that contains the published .paws.pth models used for inference. "
            "The app lists published model files found directly inside this folder so "
            "you can choose one below. Leave this at the default unless you saved your "
            "published models somewhere else."
        ),
    )
    if model_dir_input != st.session_state["model_dir"]:
        st.session_state["model_dir"] = model_dir_input
    model_dir = st.session_state["model_dir"]
    paths, err = list_published_models(model_dir)
    if err:
        if err == "Model directory does not exist yet.":
            if st.button(
                "Create Directory", icon=ICON_CREATE, key="btn_create_model_dir"
            ):
                ok, mkdir_err = safe_mkdir(model_dir)
                if ok:
                    st.rerun()
                else:
                    st.error(f"Could not create model directory: {mkdir_err}")
            else:
                st.warning(err)
        else:
            st.warning(err)
    elif not paths:
        st.info("No .paws.pth files found in this directory.")
    else:
        st.success(f"Found {len(paths)} published model file(s).")

    return paths


def render_model_metadata(metadata):
    """
    Render summary metadata for the selected model.

    Args:
        metadata (dict): Published model metadata dictionary.

    Returns:
        None: Renders model metadata boxes and detail rows.
    """
    display_name = html.escape(
        str(metadata.get("display_name", "unknown") or "unknown")
    )
    architecture = html.escape(
        str(metadata.get("architecture", "unknown") or "unknown")
    )
    scale_factor = int(metadata.get("scale_factor", 1) or 1)
    weight_source = html.escape(
        str(metadata.get("weight_source", "unknown") or "unknown")
    )
    source_config = html.escape(str(metadata.get("source_config") or "not recorded"))
    source_checkpoint = html.escape(
        str(metadata.get("source_checkpoint") or "not recorded")
    )

    st.html(
        f'<div class="model-meta-header"><span class="model-meta-kicker">Display Name</span>'
        f'<div class="model-meta-title">{display_name}</div></div>'
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.html(
            f'<div class="model-stat-box"><span class="model-stat-label">Architecture</span>'
            f'<span class="model-stat-value model-stat-value-architecture">{architecture}</span></div>'
        )
    with col2:
        st.html(
            f'<div class="model-stat-box"><span class="model-stat-label">Scale</span>'
            f'<span class="model-stat-value">x{scale_factor}</span></div>'
        )
    with col3:
        st.html(
            f'<div class="model-stat-box"><span class="model-stat-label">Weights</span>'
            f'<span class="model-stat-value">{weight_source}</span></div>'
        )

    st.html(
        '<div class="model-detail-list">'
        f'<div class="model-detail-row"><span class="model-detail-label">Source config</span>'
        f'<span class="model-detail-value">{source_config}</span></div>'
        f'<div class="model-detail-row"><span class="model-detail-label">Source checkpoint</span>'
        f'<span class="model-detail-value">{source_checkpoint}</span></div>'
        "</div>"
    )


def render_model_picker(paths):
    """
    Render the model picker and metadata preview.

    Args:
        paths (list): Published model paths.

    Returns:
        tuple: Selected model path and optional metadata dictionary.
    """
    if not paths:
        return "", None

    labels = [Path(path).name for path in paths]
    selected_label = st.selectbox("Model Checkpoint", labels, index=0)
    selected_path = paths[labels.index(selected_label)]
    metadata, err = read_model_metadata_for_ui(selected_path)
    if err:
        st.error(f"Could not read model metadata: {err}")
        return selected_path, None

    render_model_metadata(metadata)
    return selected_path, metadata


def render_tile_input(label, min_value, max_value, step, key, help, default_value):
    """
    Render one persisted tile input with clamped session state.

    Args:
        label (str): Widget label.
        min_value (int): Minimum allowed value.
        max_value (int): Maximum allowed value.
        step (int): Number input step size.
        key (str): Session-state widget key.
        help (str): Widget help text.
        default_value (int): Default value when session state is missing.

    Returns:
        int: Selected integer value.
    """
    try:
        st.session_state[key] = max(
            min_value, min(max_value, int(st.session_state.get(key, default_value)))
        )
    except Exception:
        st.session_state[key] = default_value
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        key=key,
        help=help,
    )


def render_tile_controls():
    """
    Render temporal and spatial tiling controls.

    Returns:
        dict: Tiling settings dictionary.
    """
    st.divider()
    st.subheader("Tiling")
    st.info(
        "Tiling splits the video into smaller sections to reduce GPU memory use. "
        "Smaller tile sizes are safer but usually slower. If you run out of GPU memory, "
        "lower the tile values.",
        icon=ICON_INFO,
    )

    col_t, col_h, col_w = st.columns(3)
    with col_t:
        tile_t = render_tile_input(
            "Temporal Tile Length (frames)",
            min_value=MIN_TEMPORAL_TILE,
            max_value=240,
            step=1,
            key="tile_t_input",
            help=(
                f"How many frames are processed at once. Smaller values use less GPU memory but "
                f"are usually slower. Larger values are faster when they fit in memory, and can "
                f"improve restoration by giving the model more temporal context across frames. "
                f"Minimum: {MIN_TEMPORAL_TILE} frames. Default: {DEFAULT_TEMPORAL_CHUNK} frames. "
                "Larger values are clamped to the video length."
            ),
            default_value=DEFAULT_TEMPORAL_CHUNK,
        )
    with col_h:
        tile_h = render_tile_input(
            "Spatial Tile Height (pixels)",
            min_value=MIN_SPATIAL_TILE,
            max_value=4096,
            step=16,
            key="tile_h_input",
            help=(
                f"The height of each processed section in input pixels. Smaller values use less "
                f"GPU memory but are usually slower. Higher values are faster when they fit in "
                f"memory. Minimum: {MIN_SPATIAL_TILE} pixels. Default: {DEFAULT_TILE_HEIGHT} "
                "pixels. Larger values are clamped to the input frame height."
            ),
            default_value=DEFAULT_TILE_HEIGHT,
        )
    with col_w:
        tile_w = render_tile_input(
            "Spatial Tile Width (pixels)",
            min_value=MIN_SPATIAL_TILE,
            max_value=4096,
            step=16,
            key="tile_w_input",
            help=(
                f"The width of each processed section in input pixels. Smaller values use less "
                f"GPU memory but are usually slower. Higher values are faster when they fit in "
                f"memory. Minimum: {MIN_SPATIAL_TILE} pixels. Default: {DEFAULT_TILE_WIDTH} "
                "pixels. Larger values are clamped to the input frame width."
            ),
            default_value=DEFAULT_TILE_WIDTH,
        )

    col_pt, col_ph, col_pw = st.columns(3)
    with col_pt:
        pad_t = st.number_input(
            "Temporal Padding (frames)",
            min_value=0,
            max_value=120,
            value=0,
            step=1,
            help=(
                "Extra overlap frames between temporal tiles. More padding can improve results "
                "near tile boundaries, but uses more GPU memory."
            ),
        )
    with col_ph:
        pad_h = st.number_input(
            "Spatial Padding Height (pixels)",
            min_value=0,
            max_value=512,
            value=DEFAULT_PAD_H,
            step=4,
            help=(
                "Extra overlap pixels around each tile. More padding can reduce visible seams, "
                "but uses more GPU memory."
            ),
        )
    with col_pw:
        pad_w = st.number_input(
            "Spatial Padding Width (pixels)",
            min_value=0,
            max_value=512,
            value=DEFAULT_PAD_W,
            step=4,
            help=(
                "Extra overlap pixels around each tile. More padding can reduce visible seams, "
                "but uses more GPU memory."
            ),
        )

    return {
        "tile_t": int(tile_t),
        "tile_h": int(tile_h),
        "tile_w": int(tile_w),
        "pad_t": int(pad_t),
        "pad_h": int(pad_h),
        "pad_w": int(pad_w),
    }


def render_model_specific_controls(metadata):
    """
    Render controls that depend on model metadata.

    Args:
        metadata (dict | None): Published model metadata dictionary, or None.

    Returns:
        dict: Model-specific settings dictionary.
    """
    settings = {}
    if not metadata:
        return settings

    model_kwargs = metadata.get("model_kwargs") or {}
    if "is_sequential_cleaning" not in model_kwargs:
        return settings

    architecture = metadata.get("architecture") or "selected model"
    default_value = bool(model_kwargs.get("is_sequential_cleaning", False))

    st.divider()
    st.subheader("Model Options")
    settings["is_sequential_cleaning"] = st.toggle(
        "Sequential Cleaning",
        value=default_value,
        help=f"Run the {architecture} cleaning stage one frame at a time to reduce memory.",
    )
    return settings


def render_runtime_controls():
    """
    Render general runtime controls.

    Returns:
        dict: Runtime settings dictionary.
    """
    st.divider()
    st.subheader("Runtime")

    precision_options = list(PRECISION_OPTIONS)
    default_precision = "bf16" if cuda_supports_bf16() else "fp32"
    precision = st.selectbox(
        "Precision",
        precision_options,
        index=precision_options.index(default_precision),
        help=(
            "fp32 is the safest option. It uses the most GPU memory and is the slowest. "
            "bf16 is the best balance on modern CUDA GPUs; it uses less memory than fp32 "
            "while maintaining similar results. fp16 uses the least memory but can cause "
            "visual artifacts, black frames, or blank output. Note: bf16 and fp16 require "
            "CUDA. If you see visual glitches, switch to fp32. If you run out of GPU "
            "memory, use a 16-bit mode."
        ),
    )
    compile_mode = "none"
    with st.expander("Experimental Compiler", expanded=False):
        st.info(
            "Recommendation: Keep this set to None for normal use. Compile mode can speed up very "
            "long videos, but the first frames may take several minutes while the model compiles. "
            "Different tile shapes near the edges of a video can trigger more compilation delays.",
            icon=ICON_INFO,
        )
        st.caption(
            "Requirements:\n"
            "Windows CUDA requires CUDA, MSVC, and a compatible Triton build (e.g., triton-windows).\n"
            "Linux requires a C++ toolchain; Linux CUDA also requires Triton."
        )
        compile_mode = st.selectbox("Compile Mode", list(COMPILE_MODE_OPTIONS), index=0)
        if compile_mode != "none":
            st.warning(
                "Compile mode is enabled. Startup may be slow, and the app will fall back to "
                "normal inference if compilation or compiler setup fails.",
                icon=ICON_WARNING,
            )

    return {"precision": precision, "compile_mode": compile_mode}


def render_model_settings():
    """
    Render published-model and inference controls.

    Returns:
        dict: Model inference settings dictionary.
    """
    paths = render_model_directory_controls()
    model_path, metadata = render_model_picker(paths)

    model_settings = {}
    model_settings.update(render_model_specific_controls(metadata))
    model_settings.update(render_tile_controls())
    model_settings.update(render_runtime_controls())
    model_settings["model_path"] = model_path
    model_settings["metadata"] = metadata or {}

    if metadata:
        model_settings["architecture"] = metadata.get("architecture", "")
        model_settings["scale_factor"] = int(metadata.get("scale_factor", 1) or 1)

    return model_settings
