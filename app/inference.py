"""
Coordinate end-to-end video super-resolution inference.
"""

import traceback

import av
import torch

from helpers import estimate_total_frames
from helpers import first_stream
from helpers import make_result
from helpers import quiet_close
from inference_io import close_output_writer
from inference_io import force_frame_progress
from inference_io import make_output_writer
from inference_io import write_sr_clip
from inference_runtime import build_failure_technical_details
from inference_runtime import load_inference_model
from inference_runtime import log_runtime_message
from inference_runtime import make_model_runtime
from inference_runtime import notify_runtime_status
from inference_runtime import validate_runtime_args
from inference_tiling import infer_window_with_retry
from inference_tiling import make_frame_record
from inference_tiling import resolve_runtime_tiling
from inference_tiling import resolve_temporal_chunk_size
from media import JobCancelled
from media import StreamlitProgress
from media import require_not_cancelled


def process_ready_chunks(
    state,
    model_runtime,
    model_settings,
    scale,
    device,
    writer,
    progress,
    progress_cb,
    total_frames,
    check_cancel,
    eof,
):
    """
    Run inference on every buffered chunk that is ready.

    state tracks: buffer (frame records), buffer_start (first valid index),
    decoded (frames seen), next_core (next frame to process).

    Args:
        state (dict): Mutable stream state.
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.
        model_settings (dict): Inference settings dictionary.
        scale (int): Model scale factor.
        device (torch.device): Active torch device.
        writer (dict): Output writer dictionary.
        progress (object | None): Optional foreground progress helper.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        check_cancel (callable | None): Optional cancellation function.
        eof (bool): True when input decoding has finished.

    Returns:
        None: Writes super-resolved frames to the output writer.
    """
    core_size = resolve_temporal_chunk_size(model_settings)
    pad_t = int(model_settings.get("pad_t", 0) or 0)

    while True:
        decoded_frames = state["decoded"]
        next_core_start = state["next_core"]

        # Check if there's a chunk ready to process.
        if next_core_start >= decoded_frames:
            break
        if not eof and decoded_frames < next_core_start + core_size + pad_t:
            break

        require_not_cancelled(check_cancel)
        core_start = next_core_start
        core_end = min(core_start + core_size, decoded_frames)
        # The inference window includes temporal padding on both sides, but only
        # the center core range is written to the output stream.
        window_start = max(0, core_start - pad_t)
        window_end = min(decoded_frames, core_end + pad_t)
        buffer_left = window_start - state["buffer_start"]
        buffer_right = window_end - state["buffer_start"]
        core_offset_start = core_start - window_start
        core_offset_end = core_offset_start + (core_end - core_start)

        records = state["buffer"][buffer_left:buffer_right]
        sr_clip = infer_window_with_retry(
            model_runtime,
            records,
            model_settings,
            scale,
            device,
            progress_cb,
            total_frames,
            writer.get("frame_index", 0),
        )
        core_clip = sr_clip[core_offset_start:core_offset_end]
        core_records = records[core_offset_start:core_offset_end]
        log_runtime_message(
            "upscaling",
            "writing "
            f"{len(core_records)} core frames after {len(records)}-frame padded inference window "
            f"(core={core_start}-{core_end - 1}, window={window_start}-{window_end - 1})",
        )
        write_sr_clip(
            writer,
            core_clip,
            core_records,
            progress,
            progress_cb,
            total_frames,
            check_cancel,
        )
        force_frame_progress(progress, progress_cb, writer["frame_index"], total_frames)

        state["next_core"] = core_end
        # Drop buffered frames that are no longer needed.
        keep_from = max(0, core_end - pad_t)
        drop_count = keep_from - state["buffer_start"]
        if drop_count > 0:
            state["buffer"] = state["buffer"][drop_count:]
            state["buffer_start"] = keep_from


def process_stream_packets(
    container_in,
    video_stream_in,
    audio_stream_in,
    writer,
    model_runtime,
    model_settings,
    scale,
    device,
    progress,
    progress_cb,
    total_frames,
    check_cancel,
):
    """
    Stream decoded packets through inference and output writing.

    state tracks: buffer (frame records), buffer_start (first valid index),
    decoded (frames seen), next_core (next frame to process).

    Args:
        container_in (av.container.input.InputContainer): Open input container.
        video_stream_in (object): Input video stream.
        audio_stream_in (object | None): Input audio stream, or None.
        writer (dict): Output writer dictionary.
        model_runtime (dict | torch.nn.Module): Runtime dictionary or model object.
        model_settings (dict): Inference settings dictionary.
        scale (int): Model scale factor.
        device (torch.device): Active torch device.
        progress (object | None): Optional foreground progress helper.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        check_cancel (callable | None): Optional cancellation function.

    Returns:
        int: Number of decoded frames.
    """
    state = {"buffer": [], "buffer_start": 0, "decoded": 0, "next_core": 0}
    streams = [video_stream_in]
    if audio_stream_in is not None and not writer.get("is_png"):
        streams.append(audio_stream_in)

    for packet in container_in.demux(streams):
        require_not_cancelled(check_cancel)
        if packet is None:
            continue

        # Route audio packets directly to the audio pipeline.
        if packet.stream.type == "audio" and writer.get("audio_pipe") is not None:
            writer["audio_pipe"].handle_audio_packet(packet, writer["container_out"])
            continue
        if packet.stream.type != "video":
            continue

        # Decode video frames and append to buffer.
        for frame in packet.decode():
            require_not_cancelled(check_cancel)
            if not isinstance(frame, av.VideoFrame):
                continue
            state["buffer"].append(make_frame_record(frame))
            state["decoded"] += 1

        # Process any ready chunks now that more frames are buffered.
        process_ready_chunks(
            state,
            model_runtime,
            model_settings,
            scale,
            device,
            writer,
            progress,
            progress_cb,
            total_frames,
            check_cancel,
            False,
        )

    # Process remaining frames after EOF.
    final_total = max(total_frames, state["decoded"])
    process_ready_chunks(
        state,
        model_runtime,
        model_settings,
        scale,
        device,
        writer,
        progress,
        progress_cb,
        final_total,
        check_cancel,
        True,
    )
    return state["decoded"]


def super_resolve_video(
    input_path, output_path, model_settings, out, progress_cb=None, check_cancel=None
):
    """
    Run PAWS inference for one input video.

    Args:
        input_path (str): Source video path.
        output_path (str): Destination output path.
        model_settings (dict): Inference settings dictionary.
        out (dict): Output settings dictionary.
        progress_cb (callable | None): Optional frame progress callback.
        check_cancel (callable | None): Optional cancellation function.

    Returns:
        dict: Result dictionary.
    """
    container_in = None
    writer = None
    progress = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    is_png_output = out.get("export_mode") == "PNG Image Sequence"

    try:
        validate_runtime_args(
            device,
            model_settings.get("precision", "fp32"),
            model_settings.get("compile_mode", "none"),
        )
        model, artifact = load_inference_model(
            model_settings.get("model_path"), model_settings, device
        )
        scale = int(artifact.get("scale_factor", 1) or 1)

        container_in = av.open(input_path)
        video_stream_in = first_stream(container_in, "video")
        if video_stream_in is None:
            failure_message = "No video stream found."
            return make_result(
                False,
                failure_message,
                output_path,
                is_png_output,
                technical_details=build_failure_technical_details(
                    input_path,
                    output_path,
                    model_settings,
                    out,
                    failure_message,
                ),
            )

        audio_stream_in = first_stream(container_in, "audio")
        total_frames = estimate_total_frames(container_in, video_stream_in)
        progress = StreamlitProgress(1) if progress_cb is None else None
        force_frame_progress(progress, progress_cb, 0, total_frames)

        model_settings = resolve_runtime_tiling(
            model_settings,
            video_stream_in.height,
            video_stream_in.width,
            total_frames,
        )
        notify_runtime_status(
            progress_cb, total_frames, model_settings.get("tiling_message", "")
        )
        model_runtime = make_model_runtime(
            model, model_settings, progress_cb, total_frames, 0
        )
        writer = make_output_writer(
            output_path, out, video_stream_in, audio_stream_in, scale, progress_cb
        )

        decoded = process_stream_packets(
            container_in,
            video_stream_in,
            audio_stream_in,
            writer,
            model_runtime,
            model_settings,
            scale,
            device,
            progress,
            progress_cb,
            total_frames,
            check_cancel,
        )
        is_png = writer.get("is_png")
        close_output_writer(writer)
        writer = None

        if progress is not None:
            progress.finish(decoded, max(total_frames, decoded))
            progress.close()

        return make_result(True, "Success", output_path, is_png)
    except JobCancelled:
        raise
    except Exception as exc:
        failure_message = f"Error during model inference: {exc}"
        return make_result(
            False,
            failure_message,
            output_path,
            is_png_output,
            technical_details=build_failure_technical_details(
                input_path,
                output_path,
                model_settings,
                out,
                failure_message,
                exc=exc,
                traceback_text=traceback.format_exc(),
            ),
        )
    finally:
        quiet_close(container_in)
        if writer is not None:
            try:
                close_output_writer(writer)
            except Exception:
                pass
