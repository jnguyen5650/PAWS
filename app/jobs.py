"""
Manage background inference jobs using threads.

One job at a time. No queue. Recent history kept for last 10 finished jobs.
"""

import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from constants import CONTAINER_EXT_MAP
from constants import PUBLISHED_MODEL_EXT
from helpers import ensure_output_dir
from helpers import quiet_remove_file
from helpers import write_upload_to_temp
from inference import super_resolve_video
from media import JobCancelled

MAX_RECENT_JOBS = 10

# Module-level state protected by a lock.
_STATE = {
    "lock": threading.Lock(),
    "active": None,  # Currently running job dict, or None.
    "active_thread": None,  # Thread object, or None.
    "recent": [],  # List of finished job dicts (most recent first).
}


def _build_source_summary(source_names):
    """
    Build a short summary of the selected source files.

    Args:
        source_names (list[str]): Display names of the source files.

    Returns:
        str: Single-line summary of the source files.
    """
    if not source_names:
        return "No files"
    if len(source_names) == 1:
        return source_names[0]
    return f"{source_names[0]} + {len(source_names) - 1} more"


def _build_output_path(display_name, model_settings, out):
    """
    Build the output file or folder path for one source file.

    Args:
        display_name (str): Display name of the source file.
        model_settings (dict): Model inference settings dictionary.
        out (dict): Output settings dictionary.

    Returns:
        str: Output file or folder path.
    """
    if out["export_mode"] == "PNG Image Sequence":
        return os.path.join(out["output_dir"], f"sr_{Path(display_name).stem}_frames")

    ext = CONTAINER_EXT_MAP.get(out.get("container_format") or "", ".mp4")
    scale = int(model_settings.get("scale_factor", 1) or 1)
    return os.path.join(
        out["output_dir"],
        f"paws_sr_{Path(display_name).stem}_x{scale}{ext}",
    )


def _validate_settings(model_settings, out):
    """
    Validate output and model settings before processing starts.

    Args:
        model_settings (dict): Model inference settings dictionary.
        out (dict): Output settings dictionary.

    Returns:
        str or None: Error message if validation fails, or None on success.
    """
    err = ensure_output_dir(out)
    if err:
        return err

    model_path = model_settings.get("model_path")
    if out.get("export_mode") == "Standard Video Encoding" and not out.get(
        "video_codec"
    ):
        return "No video codec selected."
    if not model_path:
        return "No published model selected."
    if not os.path.exists(model_path):
        return f"Published model not found: {model_path}"
    if not model_path.endswith(PUBLISHED_MODEL_EXT):
        return f"Expected a {PUBLISHED_MODEL_EXT} published model file."
    return None


def _cleanup_partial_output(out, output_path):
    """
    Remove partial output after cancellation.

    Args:
        out (dict): Output settings dictionary.
        output_path (str): Output file or folder path to clean up.

    Returns:
        None: Removes the partial output file or folder.
    """
    if out.get("export_mode") == "PNG Image Sequence":
        shutil.rmtree(output_path, ignore_errors=True)
    else:
        quiet_remove_file(output_path)


def _update_job(job, **kw):
    """
    Update job fields and refresh the timestamp under lock.

    Args:
        job (dict): Mutable job dictionary.
        **kw: Field names and values to set.

    Returns:
        None: Updates the job fields and timestamp.
    """
    with job["_lock"]:
        job.update(kw)
        job["updated_at"] = time.time()


def _on_progress(job, processed, total, message=None):
    """
    Called by inference to report frame progress.

    Args:
        job (dict): Mutable job dictionary.
        processed (int): Number of processed frames.
        total (int): Estimated total frame count.
        message (str | None): Optional status message.

    Returns:
        None: Updates job progress fields. Raises JobCancelled if cancelled.

    Raises:
        JobCancelled: If the cancel event is set.
    """
    if job["_cancel_event"].is_set():
        raise JobCancelled()
    with job["_lock"]:
        job["frame_processed"] = int(processed)
        job["frame_total"] = int(total)
        job["updated_at"] = time.time()
        if message:
            job["message"] = message


def _cancel(job, completed, total):
    """
    Mark the job as cancelled and log the result.

    Args:
        job (dict): Mutable job dictionary.
        completed (int): Number of completed files.
        total (int): Total number of files.

    Returns:
        None: Marks the job as cancelled and updates timestamps.
    """
    _log(f"cancelled after {completed}/{total} file(s).")
    _update_job(
        job, state="cancelled", message="Run cancelled.", completed_files=completed
    )
    job["finished_at"] = time.time()


def _make_job_dict(items, model_settings, out, temp_to_delete):
    """
    Create a new job dict for UI tracking.

    Args:
        items (list[dict]): List of input file dictionaries with display_name and input_path.
        model_settings (dict): Model inference settings dictionary.
        out (dict): Output settings dictionary.
        temp_to_delete (list[str]): List of temporary file paths to clean up.

    Returns:
        dict: Initialized job dictionary with all tracking fields.
    """
    source_names = [item["display_name"] for item in items]
    now = time.time()
    return {
        "job_id": uuid.uuid4().hex[:12],
        "state": "starting",
        "message": "Starting run...",
        "submitted_at": now,
        "finished_at": None,
        "total_files": len(items),
        "current_index": 0,
        "completed_files": 0,
        "current_name": "",
        "frame_processed": 0,
        "frame_total": 0,
        "results": [],
        "traceback": "",
        "source_summary": _build_source_summary(source_names),
        "cancel_requested": False,
        # Internal fields (not exposed to UI snapshot).
        "_lock": threading.Lock(),
        "_cancel_event": threading.Event(),
        "_temp_to_delete": list(temp_to_delete),
        "_items": list(items),
        "_model_settings": dict(model_settings),
        "_out": dict(out),
    }


def _log(message):
    """
    Print a tagged log message.

    Args:
        message (str): Message text to print.

    Returns:
        None: Prints the tagged log message.
    """
    print(f"[PAWS batch] {message}", flush=True)


def _run_job(job):
    """
    Worker thread: process all files in a job.

    Args:
        job (dict): Job dictionary with all tracking fields.

    Returns:
        None: Processes all files in the job.
    """
    items = job["_items"]
    model_settings = job["_model_settings"]
    out = job["_out"]
    cancel_event = job["_cancel_event"]
    temp_to_delete = job["_temp_to_delete"]
    results = []

    def progress_cb(processed, total, message=None, force=False):
        _on_progress(job, processed, total, message)

    try:
        settings_error = _validate_settings(model_settings, out)
        if settings_error:
            _update_job(job, state="error", message=settings_error)
            job["finished_at"] = time.time()
            return

        _update_job(job, state="running", message="Starting run...")
        total_files = len(items)

        for index, item in enumerate(items, start=1):
            if cancel_event.is_set():
                _cancel(job, len(results), total_files)
                return

            display_name = item["display_name"]
            output_path = _build_output_path(display_name, model_settings, out)
            _update_job(
                job,
                message=f"Processing file {index} of {total_files}: {display_name}",
                current_index=index,
                current_name=display_name,
                frame_processed=0,
                frame_total=0,
                completed_files=len(results),
            )

            try:
                result = super_resolve_video(
                    item["input_path"],
                    output_path,
                    model_settings,
                    out,
                    progress_cb,
                    cancel_event.is_set,
                )
            except JobCancelled:
                _cleanup_partial_output(out, output_path)
                _cancel(job, len(results), total_files)
                return

            row = dict(result)
            row["display_name"] = display_name
            results.append(row)
            _update_job(job, results=list(results), completed_files=len(results))

            if result.get("ok"):
                _log(f"Completed {index}/{total_files}: {display_name}")
            else:
                _log(
                    f"Failed {index}/{total_files}: {display_name} - {result.get('message', '')}"
                )

        succeeded = sum(1 for r in results if r.get("ok"))
        failed = len(results) - succeeded
        _log(
            f"complete: {len(results)} file(s), {succeeded} succeeded, {failed} failed."
        )
        _update_job(job, state="complete", message="Run complete.")
        job["finished_at"] = time.time()

    except Exception as exc:
        _log(f"worker error: {exc}")
        _update_job(job, state="error", message=f"Run stopped with an error: {exc}")
        job["traceback"] = traceback.format_exc()
        job["finished_at"] = time.time()
    finally:
        for temp_path in temp_to_delete:
            quiet_remove_file(temp_path)


def start_background_job(use_local, local_paths, uploads, model_settings, out):
    """
    Stage inputs and start a background thread.

    Args:
        use_local (bool): True when local-file mode is enabled.
        local_paths (list[str]): Selected local file paths.
        uploads (list): Uploaded file objects.
        model_settings (dict): Model inference settings dictionary.
        out (dict): Output settings dictionary.

    Returns:
        dict: Dictionary with job_id and state keys.

    Raises:
        RuntimeError: If a job is already running or upload staging fails.
    """
    with _STATE["lock"]:
        if _STATE["active"] is not None:
            raise RuntimeError("A job is already running. Wait for it to finish first.")

    # Stage input files.
    items = []
    temp_to_delete = []

    if use_local:
        for path in local_paths:
            items.append({"display_name": Path(path).name, "input_path": str(path)})
    else:
        for upload in uploads:
            tmp_path, err = write_upload_to_temp(upload)
            if err or not tmp_path:
                raise RuntimeError(f"Failed to stage upload '{upload.name}': {err}")
            temp_to_delete.append(tmp_path)
            items.append({"display_name": upload.name, "input_path": tmp_path})

    job = _make_job_dict(items, model_settings, out, temp_to_delete)
    thread = threading.Thread(
        target=_run_job,
        args=(job,),
        name="paws-worker",
        daemon=True,
    )

    with _STATE["lock"]:
        _STATE["active"] = job
        _STATE["active_thread"] = thread

    thread.start()
    _log(f"Job {job['job_id']} started.")
    return {"job_id": job["job_id"], "state": "starting"}


def request_job_cancel(job_id):
    """
    Request cancellation for the active job.

    Args:
        job_id (str): Job ID of the active job to cancel.

    Returns:
        bool: True when the request was accepted.
    """
    if not job_id:
        return False

    with _STATE["lock"]:
        job = _STATE["active"]
        if not job or job["job_id"] != job_id:
            return False
        if job["state"] not in ("starting", "running", "cancelling"):
            return False

        job["_cancel_event"].set()
        job["cancel_requested"] = True
        job["state"] = "cancelling"
        job["message"] = "Cancellation requested. Stopping after the current step."
        job["updated_at"] = time.time()

    _log(f"Run {job_id} cancellation requested.")
    return True


def _snapshot(job):
    """
    Create a UI-safe copy of a job dict (no internal fields).

    Args:
        job (dict or None): Job dictionary, or None.

    Returns:
        dict or None: Shallow copy of the job without internal fields, or None.
    """
    if job is None:
        return None
    return {
        "job_id": job["job_id"],
        "state": job["state"],
        "message": job["message"],
        "submitted_at": job["submitted_at"],
        "updated_at": job["updated_at"],
        "finished_at": job["finished_at"],
        "total_files": job["total_files"],
        "current_index": job["current_index"],
        "completed_files": job["completed_files"],
        "current_name": job["current_name"],
        "frame_processed": job["frame_processed"],
        "frame_total": job["frame_total"],
        "results": [dict(r) for r in job["results"]],
        "traceback": job["traceback"],
        "source_summary": job["source_summary"],
        "cancel_requested": job["cancel_requested"],
    }


def get_job_manager_snapshot():
    """
    Build the current UI snapshot.

    Checks if the active thread has finished and moves it to recent if so.

    Returns:
        dict: Snapshot with active_job and recent_jobs keys.
    """
    with _STATE["lock"]:
        active_job = _STATE["active"]
        active_thread = _STATE["active_thread"]

        # If the worker thread has exited, move the job to recent.
        if active_job is not None and active_thread is not None:
            if not active_thread.is_alive():
                _STATE["recent"].insert(0, active_job)
                while len(_STATE["recent"]) > MAX_RECENT_JOBS:
                    _STATE["recent"].pop()
                _STATE["active"] = None
                _STATE["active_thread"] = None

        return {
            "active_job": _snapshot(_STATE["active"]),
            "recent_jobs": [_snapshot(j) for j in _STATE["recent"]],
        }


def clear_recent_jobs():
    """
    Clear finished-job history.

    Returns:
        None: Clears the recent jobs list.
    """
    with _STATE["lock"]:
        _STATE["recent"].clear()
