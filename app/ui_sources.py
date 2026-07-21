"""
Render source selection and preview metadata controls.
"""

import html
from pathlib import Path

import streamlit as st

from constants import ICON_FOLDER
from helpers import get_video_info
from helpers import list_videos_in_folder
from helpers import quiet_remove_file
from helpers import write_upload_to_temp


def preview_metric_items(info):
    """
    Build preview metric rows from video metadata.

    Args:
        info (dict): Video metadata dictionary.

    Returns:
        list: ``(label, value)`` pairs for the preview row.
    """
    return [
        ("Resolution", f'{info["width"]} x {info["height"]}'),
        ("Duration", f'{int(info["duration_sec"])}s'),
        ("FPS", f'{info["fps"]:.2f}'),
        ("Codec", info["codec"]),
    ]


def render_metric_boxes(items):
    """
    Render preview metric boxes in a single row.

    Args:
        items (list[tuple]): Sequence of ``(label, value)`` pairs.

    Returns:
        None: Renders metric boxes in a single row.
    """
    for col, item in zip(st.columns(len(items)), items):
        with col:
            st.html(
                f'<div class="metric-box"><span class="stat-label">{html.escape(item[0])}</span>'
                f'<span class="stat-value">{html.escape(item[1])}</span></div>'
            )


def render_preview_for_path(file_path, show_header_on_error):
    """
    Read and render preview metadata for one video path.

    Args:
        file_path (str): Path to the preview video.
        show_header_on_error (bool): True to show the header even on metadata failure.

    Returns:
        None: Renders the preview metadata for one video path.
    """
    info, err = get_video_info(file_path)
    if show_header_on_error or info:
        st.divider()
        st.subheader("Preview Metadata (First File)")
    if err:
        st.warning(f"Could not read metadata for preview: {err}")
        return
    if info:
        render_metric_boxes(preview_metric_items(info))


def render_local_source_controls():
    """
    Render local-folder source controls.

    Returns:
        list: Selected local paths.
    """
    input_dir = st.text_input(
        "Local folder path containing videos",
        placeholder=str(Path.home() / "Videos"),
        key="local_folder_path",
        help=(
            "Enter the full path to the folder that contains the source videos you want to "
            "process. The app lists supported video files found directly inside this folder. "
            "Subfolders are ignored."
        ),
    )
    paths, err = list_videos_in_folder(input_dir)
    if err:
        if "Permission denied" in err:
            st.error(err)
        else:
            st.warning(err)
        return []
    if not paths and input_dir:
        st.info("No video files found in that folder.")
        return []
    if not paths:
        return []

    st.success(f"Found {len(paths)} video file(s).")
    local_paths = st.multiselect(
        "Select videos to process",
        paths,
        default=paths[:3] if len(paths) > 3 else paths,
        help=(
            "Choose which files from this folder should be processed in this job. You can "
            "search by filename in this list. The first selected file is also used for the "
            "preview metadata shown below."
        ),
    )
    if local_paths:
        render_preview_for_path(local_paths[0], False)
    return local_paths


def render_upload_source_controls():
    """
    Render uploaded-file source controls.

    Returns:
        list: Uploaded file objects.
    """
    st.caption(
        "Upload one or more videos here. This is best for smaller files. For uploads around "
        "200MB or larger, or for larger batches, switch to **Local Folder** mode to process "
        "files directly from disk."
    )
    uploads = (
        st.file_uploader(
            "Drag & Drop Videos Here (MP4, MOV, AVI, WEBM)",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        or []
    )
    if not uploads:
        return uploads

    if sum(file_obj.size for file_obj in uploads) / (1024 * 1024) > 190:
        st.warning(
            "Total upload size is large. Switch to **Local Folder** mode for faster, more "
            "reliable processing."
        )
    st.success(f"Selected {len(uploads)} video file(s)")

    tmp_path, err = write_upload_to_temp(uploads[0])
    if err:
        st.warning(f"Could not read metadata for preview: {err}")
        return uploads

    try:
        render_preview_for_path(tmp_path, True)
    finally:
        quiet_remove_file(tmp_path)
    return uploads


def render_upload_section():
    """
    Render source selection controls and preview metadata.

    Returns:
        tuple: Mode flag, local paths, and uploads.
    """
    st.header(f"{ICON_FOLDER} Source Files")
    use_local_files = st.toggle(
        "Use local folder (for multi-GB videos)",
        value=False,
        help=(
            "Turn this on to read videos directly from a folder on the machine running this "
            "app instead of uploading them through the browser. This is faster for very large "
            "files or large batches and avoids creating temporary uploaded copies."
        ),
    )

    if use_local_files:
        local_paths = render_local_source_controls()
        return use_local_files, local_paths, []

    uploads = render_upload_source_controls()
    return use_local_files, [], uploads
