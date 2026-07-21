"""
Render job status and run results.
"""

import html
import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from constants import ICON_BROOM
from constants import ICON_CHECK
from constants import ICON_ERROR
from constants import ICON_FOLDER
from constants import ICON_SAVE
from constants import ICON_STOP
from constants import ICON_THREADS
from constants import ICON_WARNING
from constants import MIME_MAP
from jobs import clear_recent_jobs
from jobs import get_job_manager_snapshot
from jobs import request_job_cancel
from jobs import start_background_job

RUNNING_STATES = ("starting", "running", "cancelling")
FINISHED_STATES = ("complete", "cancelled", "error")
STATE_LABELS = {
    "starting": "Starting",
    "running": "Processing",
    "cancelling": "Cancelling",
    "complete": "Finished",
    "cancelled": "Cancelled",
    "error": "Error",
    "unknown": "Unknown",
}


def maybe_offer_download(output_path, idx_key):
    """
    Offer a download button for smaller output files.

    Args:
        output_path (str): Path to the output file.
        idx_key (str): Unique key suffix for the download button widget.

    Returns:
        None: Renders a download button or path info message.
    """
    try:
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
    except Exception:
        st.info(f"Saved to: `{output_path}`")
        return

    if size_mb > 200:
        st.info(f"Saved to: `{output_path}` (Too large to stream)", icon=ICON_SAVE)
        return

    try:
        with open(output_path, "rb") as file_obj:
            st.download_button(
                label=f"Download {Path(output_path).name}",
                data=file_obj.read(),
                file_name=Path(output_path).name,
                key=f"dl_{idx_key}",
                mime=MIME_MAP.get(
                    Path(output_path).suffix.lower(),
                    "application/octet-stream",
                ),
            )
    except Exception:
        st.info(f"Saved to: `{output_path}`")


def get_effective_job_state(job):
    """
    Return the effective state label for display.

    Args:
        job (dict): Job dictionary with state and cancel_requested keys.

    Returns:
        str: Human-readable state label.
    """
    state = job.get("state", "unknown")
    if job.get("cancel_requested") and state in ("starting", "running"):
        state = "cancelling"
    return STATE_LABELS.get(state, "Unknown")


def format_timestamp(timestamp):
    """
    Format a job timestamp for display.

    Args:
        timestamp (float or None): Unix timestamp, or None.

    Returns:
        str: Formatted timestamp string, or empty string if invalid.
    """
    if not timestamp:
        return ""
    try:
        return datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def render_job_header(
    job, state_label, title="Current Run", show_source=True, show_message=True
):
    """
    Render the header content for a job card.

    Args:
        job (dict): Job dictionary.
        state_label (str): Human-readable state label.
        title (str): Section title to display.
        show_source (bool): If True, show source file summary.
        show_message (bool): If True, show job message.

    Returns:
        None: Renders the job header with status, source, and message.
    """
    st.subheader(title)
    st.markdown(f"**Status:** {state_label}")
    source_summary = job.get("source_summary", "")
    if show_source and source_summary:
        st.caption(f"Source Files: {source_summary}")
    message = job.get("message", "")
    if show_message and message:
        st.write(message)


def render_active_job_panel(job):
    """
    Render the active job section with progress and controls.

    Args:
        job (dict or None): Active job dictionary, or None.

    Returns:
        None: Renders the active job panel with progress bars and controls.
    """
    if not job:
        return

    state_label = get_effective_job_state(job)
    render_job_header(
        job, state_label, title="Current Run", show_source=False, show_message=False
    )

    # Current file label.
    current_name = job.get("current_name", "")
    current_index = int(job.get("current_index", 0) or 0)
    total_files = int(job.get("total_files", 0) or 0)
    if current_name:
        if current_index > 0 and total_files > 0:
            st.caption(f"Current File: {current_index}/{total_files} - {current_name}")
        else:
            st.caption(f"Current File: {current_name}")

    # Source summary (only if useful).
    source_summary = str(job.get("source_summary") or "").strip()
    if source_summary and (total_files > 1 or source_summary != current_name):
        st.caption(f"Source Files: {source_summary}")

    # Message (skip if it duplicates the current-file row).
    message = str(job.get("message") or "").strip()
    if current_name and current_index > 0 and total_files > 0:
        expected = f"Processing file {current_index} of {total_files}: {current_name}"
        if message == expected:
            message = ""
    if message:
        st.write(message)

    # Progress bars.
    completed_files = int(job.get("completed_files", 0) or 0)
    frame_processed = int(job.get("frame_processed", 0) or 0)
    frame_total = int(job.get("frame_total", 0) or 0)
    raw_state = job.get("state", "unknown")
    if job.get("cancel_requested") and raw_state in ("starting", "running"):
        raw_state = "cancelling"

    if total_files > 0:
        st.markdown(f"**Files:** {completed_files} / {total_files}")
        progress = completed_files / total_files
        if raw_state in RUNNING_STATES and frame_total > 0:
            progress = (
                completed_files + min(frame_processed / max(frame_total, 1), 1.0)
            ) / total_files
        if raw_state == "complete":
            progress = 1.0
        st.progress(min(max(progress, 0.0), 1.0))

    if frame_total > 0 or raw_state in RUNNING_STATES:
        frame_total_text = frame_total if frame_total > 0 else "?"
        st.markdown(f"**Frames:** {frame_processed} / {frame_total_text}")
        if frame_total > 0:
            st.progress(min(frame_processed / max(frame_total, 1), 1.0))
        else:
            st.progress(0.0)

    # Cancel button.
    job_id = job.get("job_id", "")
    disabled = raw_state not in RUNNING_STATES or job.get("cancel_requested")
    with st.container(key="active-job-controls"):
        clicked = st.button(
            "Cancel Run",
            icon=ICON_STOP,
            use_container_width=True,
            disabled=disabled,
            key=f"btn_cancel_job_{job_id}",
        )
        if clicked:
            if request_job_cancel(job_id):
                st.toast("Stopping current run", icon=ICON_STOP)
                st.rerun()
            else:
                st.warning(
                    "This run already finished or is no longer active.",
                    icon=ICON_WARNING,
                )

    # Error details.
    if raw_state == "error" or job.get("traceback"):
        lines = [
            "Run",
            f"  Run ID: {job_id or 'n/a'}",
            f"  Status: {state_label}",
        ]
        if current_name:
            lines.append(f"  Current File: {current_name}")
        msg = job.get("message")
        if msg:
            lines.extend(["", "Message", msg])
        tb = (job.get("traceback") or "").rstrip()
        if tb:
            lines.extend(["", "Traceback", tb])
        with st.expander("Technical Details", expanded=False):
            st.code("\n".join(lines), language="text")

    # Results.
    render_job_results(job)

    # Footer timestamps.
    render_job_footer(job)


def render_job_footer(job):
    """
    Render low-priority run metadata in the footer.

    Args:
        job (dict): Job dictionary with timestamps and state.

    Returns:
        None: Renders the job footer with timestamps and run ID.
    """
    submitted = format_timestamp(job.get("submitted_at"))
    finished = format_timestamp(job.get("finished_at"))
    state = job.get("state")
    job_id = job.get("job_id", "")

    rows = []
    if submitted:
        rows.append(
            f'<div class="run-footer-meta-item">Submitted: {html.escape(submitted)}</div>'
        )
    if finished and state in FINISHED_STATES:
        rows.append(
            f'<div class="run-footer-meta-item">Finished: {html.escape(finished)}</div>'
        )

    if not rows and not job_id:
        return

    left = ""
    if rows:
        left = f'<div class="run-footer-cell"><div class="run-footer-meta">{"".join(rows)}</div></div>'

    right = ""
    if job_id:
        right = (
            '<div class="run-footer-cell run-footer-cell-id">'
            '<span class="run-footer-id-label">Run ID:</span> '
            f'<span class="run-footer-id-value">{html.escape(job_id)}</span>'
            "</div>"
        )

    st.html(f'<div class="run-footer-row">{left}{right}</div>')


def render_job_results(job):
    """
    Render all available results for a job.

    Args:
        job (dict): Job snapshot dictionary.

    Returns:
        None: Renders job results with download buttons or error messages.
    """
    results = job.get("results") or []
    if not results:
        return

    job_id = job.get("job_id", "")
    st.divider()
    st.subheader("Results")
    for result in results:
        name = result.get("display_name", "file")
        if result.get("ok"):
            st.success(f"{name} finished", icon=ICON_CHECK)
            if result.get("is_png"):
                st.info(
                    f"PNG frames saved in:\n`{result.get('output_path', '')}`",
                    icon=ICON_FOLDER,
                )
            else:
                maybe_offer_download(
                    result.get("output_path", ""),
                    f"bg_{job_id}_{name}",
                )
        else:
            st.error(
                f"{name} could not be processed: {result.get('message', '')}",
                icon=ICON_ERROR,
            )


def render_recent_jobs(recent_jobs):
    """
    Render recent finished, cancelled, and errored jobs.

    Args:
        recent_jobs (list[dict]): List of recent job dictionaries.

    Returns:
        None: Renders recent job cards with expandable details.
    """
    if not recent_jobs:
        return

    col_a, col_b = st.columns([3, 2])
    st.divider()
    with col_a:
        st.subheader("Recent Runs")
    with col_b:
        clicked = st.button(
            "Clear Finished Jobs",
            icon=ICON_BROOM,
            use_container_width=True,
            key="btn_clear_recent_jobs",
        )
        if clicked:
            clear_recent_jobs()
            st.rerun()

    for job in recent_jobs:
        state_label = get_effective_job_state(job)
        title = f"{state_label} - {job.get('source_summary', 'No files')}"
        with st.expander(title, expanded=state_label in ("Cancelled", "Error")):
            render_job_header(job, state_label, title="Run Details")
            render_job_results(job)
            render_job_footer(job)


def render_process_hint(snapshot):
    """
    Render the process-panel hint for the current state.

    Args:
        snapshot (dict): Job manager snapshot with active_job and recent_jobs keys.

    Returns:
        None: Renders a contextual info message about the current state.
    """
    active_job = snapshot.get("active_job")
    recent_jobs = snapshot.get("recent_jobs") or []

    if active_job:
        st.info(
            "A run is in progress. You can submit another run after this one finishes."
        )
        return

    if recent_jobs:
        st.info("No run is active. Recent runs remain below until you clear them.")
        return

    st.info(
        "Ready to start. Select your videos and model, then click **Run Inference**."
    )


@st.fragment(run_every=1)
def render_job_panel():
    """
    Render the active and recent jobs.

    Returns:
        None: Renders the job panel with active and recent job cards.
    """
    snapshot = get_job_manager_snapshot()
    render_process_hint(snapshot)
    render_active_job_panel(snapshot.get("active_job"))
    render_recent_jobs(snapshot.get("recent_jobs") or [])


def handle_process_click(
    use_local, local_paths, uploads, model_settings, output_settings
):
    """
    Start processing when the current inputs are valid.

    Args:
        use_local (bool): True when local-file mode is enabled.
        local_paths (list[str]): Selected local file paths.
        uploads (list): Uploaded file objects.
        model_settings (dict): Model inference settings dictionary.
        output_settings (dict): Output settings dictionary.

    Returns:
        None: Starts a background job or shows validation warnings.
    """
    has_sources = bool(use_local and local_paths) or bool((not use_local) and uploads)
    if not has_sources:
        st.warning(
            "Select at least one video before starting inference.", icon=ICON_WARNING
        )
        return

    if not model_settings.get("model_path"):
        st.warning("Select a model before starting inference.", icon=ICON_WARNING)
        return

    try:
        start_background_job(
            use_local,
            local_paths,
            uploads,
            model_settings,
            output_settings,
        )
    except Exception as exc:
        st.error(f"Could not start processing: {exc}", icon=ICON_ERROR)
        return

    st.toast("Inference started", icon=ICON_THREADS)
    st.rerun()
