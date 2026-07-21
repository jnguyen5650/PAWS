"""
Handle output writers and encoded-frame writes for inference.
"""

import os

import av
import cv2

from helpers import quiet_close
from helpers import safe_mkdir
from media import AudioPipeline
from media import add_video_output_stream
from media import configure_video_output_stream
from media import require_not_cancelled
from media import streamlit_notify
from media import update_frame_progress


def make_output_writer(
    output_path, out, video_stream_in, audio_stream_in, scale, progress_cb
):
    """
    Create an output writer for PNG or video export.

    Args:
        output_path (str): Output path.
        out (dict): Output settings dictionary.
        video_stream_in (object): Input video stream.
        audio_stream_in (object): Input audio stream, or None.
        scale (int): Model scale factor.
        progress_cb (callable | None): Optional background progress callback.

    Returns:
        dict: Writer dictionary.

    Raises:
        RuntimeError: If output setup fails.
    """
    if out.get("export_mode") == "PNG Image Sequence":
        ok, err = safe_mkdir(output_path)
        if not ok:
            raise RuntimeError(f"Could not create frame folder: {err}")
        return {
            "is_png": True,
            "output_path": output_path,
            "container_out": None,
            "stream_out_v": None,
            "audio_pipe": None,
            "frame_index": 0,
        }

    if not out.get("video_codec"):
        raise RuntimeError("No video codec selected.")

    container_out = av.open(output_path, mode="w")
    target_w = int(video_stream_in.width) * int(scale)
    target_h = int(video_stream_in.height) * int(scale)
    stream_out_v = add_video_output_stream(
        container_out, video_stream_in, target_w, target_h, out["video_codec"]
    )
    stream_error = configure_video_output_stream(
        stream_out_v,
        out["video_codec"],
        out["encoding_mode"],
        out["crf_value"],
        out["bitrate_kbps"],
        out["preset"],
    )
    if stream_error:
        quiet_close(container_out)
        raise RuntimeError(stream_error)

    notify = streamlit_notify if progress_cb is None else None
    audio_pipe = AudioPipeline(
        container_out, audio_stream_in, output_path, notify=notify
    )
    return {
        "is_png": False,
        "output_path": output_path,
        "container_out": container_out,
        "stream_out_v": stream_out_v,
        "audio_pipe": audio_pipe,
        "frame_index": 0,
    }


def close_output_writer(writer):
    """
    Flush and close any active output writer resources.

    Args:
        writer (dict | None): Writer dictionary from ``make_output_writer``.

    Returns:
        None: Flushes pending frames and closes the output container.
    """
    if writer is None or writer.get("is_png") or writer.get("closed"):
        return

    container_out = writer.get("container_out")
    stream_out_v = writer.get("stream_out_v")
    audio_pipe = writer.get("audio_pipe")

    try:
        if audio_pipe is not None:
            audio_pipe.flush(container_out)
        if stream_out_v is not None:
            for output_packet in stream_out_v.encode():
                container_out.mux(output_packet)
    finally:
        quiet_close(container_out)
        writer["closed"] = True


def encode_video_frame(writer, rgb_image, record):
    """
    Encode one RGB frame into the output video.

    Args:
        writer (dict): Writer dictionary.
        rgb_image (object): RGB uint8 image.
        record (dict): Source frame timing record.

    Returns:
        None: Encodes the frame and muxes it to the output container.
    """
    new_frame = av.VideoFrame.from_ndarray(rgb_image, format="rgb24")
    if record.get("pts") is not None:
        new_frame.pts = record.get("pts")
    if record.get("time_base") is not None:
        new_frame.time_base = record.get("time_base")

    stream_out_v = writer["stream_out_v"]
    container_out = writer["container_out"]
    for output_packet in stream_out_v.encode(new_frame):
        container_out.mux(output_packet)
    writer["frame_index"] += 1


def write_png_frame(writer, rgb_image):
    """
    Write one RGB frame as a numbered PNG file.

    Args:
        writer (dict): Writer dictionary.
        rgb_image (object): RGB uint8 image.

    Returns:
        None: Writes the frame as a numbered PNG file.
    """
    frame_path = os.path.join(
        writer["output_path"],
        f"{writer['frame_index']:08d}.png",
    )
    bgr_image = rgb_image[:, :, ::-1].copy()
    cv2.imwrite(frame_path, bgr_image)
    writer["frame_index"] += 1


def write_sr_clip(
    writer, sr_clip, records, progress, progress_cb, total_frames, check_cancel
):
    """
    Write a super-resolved clip to the chosen output.

    Args:
        writer (dict): Writer dictionary.
        sr_clip (object): Super-resolved clip on CPU.
        records (list): Source frame records for the clip.
        progress (object | None): Optional foreground progress helper.
        progress_cb (callable | None): Optional background progress callback.
        total_frames (int): Estimated total frame count.
        check_cancel (callable | None): Optional cancellation function.

    Returns:
        None: Writes each frame to the output writer and updates progress.
    """
    for index, sr_frame in enumerate(sr_clip):
        require_not_cancelled(check_cancel)
        tensor = sr_frame.detach().cpu().clamp(0, 1).mul(255).round().byte()
        rgb_image = tensor.permute(1, 2, 0).contiguous().numpy()
        if writer.get("is_png"):
            write_png_frame(writer, rgb_image)
        else:
            encode_video_frame(writer, rgb_image, records[index])
        update_frame_progress(
            progress, progress_cb, writer["frame_index"], total_frames
        )


def force_frame_progress(progress, progress_cb, processed, total):
    """
    Force an exact frame-progress update.

    Args:
        progress (object | None): Optional foreground progress helper.
        progress_cb (callable | None): Optional background progress callback.
        processed (int): Number of written frames.
        total (int): Estimated total frame count.

    Returns:
        None: Forces a progress update through the callback or foreground widgets.
    """
    if progress_cb is not None:
        try:
            progress_cb(processed, total, None, True)
        except TypeError:
            progress_cb(processed, total)
    elif progress is not None:
        progress.update(processed, total)
