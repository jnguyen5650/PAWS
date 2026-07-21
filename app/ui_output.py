"""
Render export format, codec, and output controls.
"""

import os

import streamlit as st

from constants import CONTAINER_TO_CODECS_UI
from constants import DEFAULT_OUT_DIR
from constants import ICON_CHECK
from constants import ICON_CREATE
from constants import ICON_INFO
from constants import ICON_TOOLS
from constants import ICON_WARNING
from helpers import crf_quality_assessment
from helpers import parse_codec_name
from helpers import safe_mkdir

EXPORT_MODE_VIDEO = "Standard Video Encoding"
EXPORT_MODE_PNG = "PNG Image Sequence"


def render_container_codec_controls():
    """
    Render container and codec controls for video export.

    Returns:
        tuple: Selected container label and codec name.
    """
    st.subheader("Container & Codec")
    col1, _col2 = st.columns(2)
    with col1:
        container_format = st.selectbox(
            "Container Format",
            ["MP4 (H.264, H.265)", "WEBM (VP9)", "MKV (Matroska)"],
            index=0,
            help=(
                "MP4 is the safest default for general use and broad playback compatibility. "
                "WEBM is mainly for VP9 web delivery. MKV is more flexible, works with all "
                "codec options in this app, and is more tolerant of interrupted exports, but "
                "MP4 is usually the better choice for phones, browsers, and general playback."
            ),
        )

    codec_options = CONTAINER_TO_CODECS_UI.get(container_format or "", [])
    codec_ui = st.selectbox(
        "Video Codec",
        codec_options,
        index=0 if codec_options else 0,
        help=(
            "H.264 is the safest default for playback compatibility. H.265 usually gives "
            "smaller files at similar quality, but some devices need newer hardware or software "
            "for smooth playback. VP9 is useful for web delivery, but it takes significantly "
            "longer to encode than H.264 or H.265."
        ),
    )
    return container_format, parse_codec_name(codec_ui)


def render_crf_controls(video_codec):
    """
    Render CRF quality controls.

    Args:
        video_codec (str): Selected output video codec.

    Returns:
        int: Selected CRF value.
    """
    st.subheader("Quality Configuration")
    col_crf1, col_crf2 = st.columns([2, 1])
    with col_crf1:
        max_crf = 63 if video_codec == "libvpx-vp9" else 51
        default_crf = 32 if video_codec == "libvpx-vp9" else 20
        crf_value = st.slider(
            "CRF Value",
            min_value=0,
            max_value=max_crf,
            value=default_crf,
            step=1,
            help=(
                "Sets a constant quality level. Lower CRF gives higher quality and larger files. "
                "Higher CRF gives smaller files and lower quality. CRF 0 is lossless and can create "
                "extremely large files. Good starting ranges: H.264 and H.265 are often best around "
                "18 to 23, and VP9 often starts around 28 to 35."
            ),
        )
    with col_crf2:
        st.metric("Quality Assessment", crf_quality_assessment(video_codec, crf_value))
    return crf_value


def render_bitrate_controls():
    """
    Render bitrate controls for VBR and CBR modes.

    Returns:
        int: Selected bitrate in kbps.
    """
    st.subheader("Bitrate Configuration")
    return st.number_input(
        "Target Bitrate (kbps)",
        min_value=50,
        max_value=100000,
        value=5000,
        step=100,
        help=(
            "Higher bitrate improves quality but increases file size. Lower bitrate creates "
            "smaller files but may introduce compression artifacts. Use this with VBR or CBR "
            "modes."
        ),
    )


def render_encoding_strategy_controls(video_codec):
    """
    Render encoding strategy controls.

    Args:
        video_codec (str): Selected output video codec.

    Returns:
        tuple: Encoding mode, CRF value, and bitrate value.
    """
    st.subheader("Encoding Strategy")
    strategy_ui = st.radio(
        "Select Encoding Mode",
        ["CRF (Best Quality)", "VBR (Balanced File Size)", "CBR (Streaming Standard)"],
        horizontal=True,
        help=(
            "CRF is the best general choice and targets visual quality. VBR balances quality "
            "and file size around a target bitrate. CBR uses a fixed bitrate and is mainly "
            "useful when a platform or workflow requires it."
        ),
    )
    encoding_mode = strategy_ui.split()[0]
    if encoding_mode == "CRF":
        return encoding_mode, render_crf_controls(video_codec), None
    return encoding_mode, 20, render_bitrate_controls()


def render_preset_control():
    """
    Render the encoder speed preset control.

    Returns:
        str: Selected preset string.
    """
    st.subheader("Encoding Speed")
    return st.selectbox(
        "Preset",
        [
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        index=5,
        help=(
            "Controls encoding speed, not AI upscaling speed. Slower presets take longer to "
            "encode but usually create smaller files at similar quality. Medium is a good "
            "default for most users."
        ),
    )


def render_output_directory_controls():
    """
    Render output-directory controls and sync session state.

    Returns:
        str: Current output directory path.
    """
    st.divider()
    st.subheader("Output Location")
    output_dir_input = st.text_input(
        "Output Directory Path",
        value=st.session_state.output_dir,
        placeholder=DEFAULT_OUT_DIR,
        help=(
            "Folder where the output video or PNG frames will be saved. The app can create "
            "this folder if it does not exist. Choose a location with enough free disk space "
            "for the finished output."
        ),
    )
    if output_dir_input != st.session_state["output_dir"]:
        st.session_state["output_dir"] = output_dir_input
    output_dir = st.session_state["output_dir"]
    if os.path.exists(output_dir):
        st.success("Directory Valid", icon=ICON_CHECK)
    else:
        if st.button("Create Directory", icon=ICON_CREATE, key="btn_create_dir"):
            ok, err = safe_mkdir(output_dir)
            if ok:
                st.rerun()
            else:
                st.error(f"Could not create directory: {err}")
        else:
            st.warning("Path does not exist yet", icon=ICON_WARNING)
    return output_dir


def build_output_settings(
    export_mode,
    output_dir,
    container_format,
    video_codec,
    encoding_mode,
    crf_value,
    bitrate_kbps,
    preset,
):
    """
    Build the output settings dictionary.

    Args:
        export_mode (str): Selected export mode label.
        output_dir (str): Output directory path.
        container_format (str | None): Selected container label, or None.
        video_codec (str): Selected codec name.
        encoding_mode (str): ``CRF``, ``VBR``, or ``CBR``.
        crf_value (int): CRF value.
        bitrate_kbps (int | None): Bitrate value, or None.
        preset (str): Encoder preset.

    Returns:
        dict: Output settings dictionary.
    """
    return {
        "export_mode": export_mode,
        "output_dir": output_dir,
        "container_format": container_format,
        "video_codec": video_codec,
        "encoding_mode": encoding_mode,
        "crf_value": int(crf_value),
        "bitrate_kbps": int(bitrate_kbps) if bitrate_kbps is not None else None,
        "preset": preset,
    }


def render_output_settings():
    """
    Render the output controls and return selected values.

    Returns:
        dict: Output settings dictionary.
    """
    st.header(f"{ICON_TOOLS} Export Settings")

    export_mode = st.selectbox(
        "Export Mode",
        [EXPORT_MODE_VIDEO, EXPORT_MODE_PNG],
        index=0,
        help=(
            "Standard Video Encoding saves one compressed video file. PNG Image Sequence saves "
            "one PNG file per frame in a folder. Choose PNG only if you need lossless frames "
            "or want to edit or re-encode them later."
        ),
    )

    container_format = None
    video_codec = ""
    encoding_mode = "CRF"
    crf_value = 20
    bitrate_kbps = None
    preset = "medium"

    if export_mode == EXPORT_MODE_VIDEO:
        container_format, video_codec = render_container_codec_controls()
        encoding_mode, crf_value, bitrate_kbps = render_encoding_strategy_controls(
            video_codec
        )
        preset = render_preset_control()
    else:
        st.info("PNG Sequence Mode Selected", icon=ICON_INFO)
        st.write(
            "This saves one PNG file per frame in a new folder instead of one video file. "
            "Use it when you need lossless frames or want to edit or re-encode them later."
        )

    output_dir = render_output_directory_controls()
    if export_mode == EXPORT_MODE_PNG:
        video_codec = "png sequence"

    return build_output_settings(
        export_mode,
        output_dir,
        container_format,
        video_codec,
        encoding_mode,
        crf_value,
        bitrate_kbps,
        preset,
    )
