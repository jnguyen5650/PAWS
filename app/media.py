"""
Handle progress updates, audio copy/re-encode, and media helpers.
"""

from pathlib import Path

import av
import streamlit as st


class JobCancelled(Exception):
    """
    Raised when a background inference run is cancelled.
    """


class StreamlitProgress:
    """
    Track foreground progress with Streamlit widgets.
    """

    def __init__(self, throttle_every_n):
        """
        Create the Streamlit progress widgets.

        Args:
            throttle_every_n (int): Draw updates every N frames.

        Returns:
            None: Initializes the progress bar and text widgets.
        """
        self._bar = st.progress(0)
        self._text = st.empty()
        self._throttle = max(1, int(throttle_every_n))

    def update(self, processed, total):
        """
        Update the foreground progress widgets.

        Args:
            processed (int): Number of processed frames.
            total (int): Estimated total frame count.

        Returns:
            None: Updates the progress bar and text widgets.
        """
        if processed % self._throttle != 0 and processed != total:
            return

        if total > 0:
            pct = min(processed / total, 1.0)
            self._text.text(
                f"Processing Frames: {processed} / {total} ({int(pct * 100)}%)"
            )
            self._bar.progress(pct)
            return

        self._text.text(f"Processing Frames: {processed} / ?")
        self._bar.progress((processed % 100) / 100.0)

    def finish(self, processed, total):
        """
        Mark the foreground progress widgets complete.

        Args:
            processed (int): Number of processed frames.
            total (int): Estimated total frame count.

        Returns:
            None: Marks the progress widgets as complete.
        """
        if total > 0:
            self._text.text(f"Processing Frames: {processed} / {total} (100%)")
        else:
            self._text.text(f"Processing Frames: {processed} / ? (done)")
        self._bar.progress(1.0)

    def close(self):
        """
        Clear the foreground progress widgets.

        Returns:
            None: Clears the progress bar and text widgets.
        """
        self._bar.empty()
        self._text.empty()


def streamlit_notify(level, message, icon=None):
    """
    Emit a Streamlit status message.

    Args:
        level (str): Streamlit method name.
        message (str): Message text.
        icon (str | None): Optional icon string.

    Returns:
        None: Emits the status message to Streamlit.
    """
    emit = getattr(st, level, None)
    if emit is None:
        return

    if level == "caption":
        emit(message)
        return

    if icon is None:
        emit(message)
    else:
        emit(message, icon=icon)


class AudioPipeline:
    """
    Handle direct audio copy or fallback audio re-encoding.
    """

    def __init__(self, container_out, audio_stream_in, output_path, notify=None):
        """
        Initialize audio handling for one export.

        Args:
            container_out (av.container.output.OutputContainer): Output container.
            audio_stream_in (object): Input audio stream, or None.
            output_path (str): Output path used to choose fallback codecs.
            notify (callable | None): Optional status callback.

        Returns:
            None: Initializes the audio output stream and resampler.
        """
        self.stream_out = None
        self.copy_mode = False
        self.resampler = None
        self.notify = notify

        if audio_stream_in is None:
            return

        # Prefer direct stream copy when the output container accepts it, and
        # only fall back to re-encode when remuxing cannot be set up.
        try:
            self.stream_out = container_out.add_stream(template=audio_stream_in)
            self.copy_mode = True
            if self.notify:
                self.notify("info", "Audio copied directly.")
            return
        except Exception:
            if self.notify:
                self.notify(
                    "warning", "Direct audio copy failed. Attempting re-encode."
                )

        # Choose fallback codec based on output container type.
        ext = Path(output_path).suffix.lower()
        if ext in (".webm", ".mkv"):
            codec_name, bit_rate, caption = (
                "libopus",
                128_000,
                "Re-encoding audio as OPUS at 128 kbps.",
            )
        else:
            codec_name, bit_rate, caption = (
                "aac",
                160_000,
                "Re-encoding audio as AAC at 160 kbps.",
            )
        if self.notify:
            self.notify("caption", caption)

        try:
            input_rate = int(audio_stream_in.rate) if audio_stream_in.rate else 48000
            self.stream_out = container_out.add_stream(codec_name, rate=input_rate)
            self.stream_out.bit_rate = bit_rate

            if audio_stream_in.layout is not None:
                self.stream_out.layout = audio_stream_in.layout.name
            elif audio_stream_in.channels:
                self.stream_out.layout = (
                    "stereo" if audio_stream_in.channels >= 2 else "mono"
                )

            # Re-encode mode normalizes layout and rate up front so decoded audio
            # frames can be handed to the encoder in a consistent format.
            target_layout = "stereo"
            if self.stream_out.layout:
                target_layout = self.stream_out.layout.name
            target_rate = self.stream_out.rate or input_rate
            self.resampler = av.audio.resampler.AudioResampler(
                format="fltp",
                layout=target_layout,
                rate=target_rate,
            )
        except Exception as exc:
            if self.notify:
                self.notify(
                    "error",
                    f"Failed to set up audio re-encode. Output will be silent. Error: {exc}",
                )
            self.stream_out = None
            self.copy_mode = False
            self.resampler = None

    def handle_audio_packet(self, packet, container_out):
        """
        Process one demuxed audio packet.

        Args:
            packet (av.packet.Packet): Demuxed audio packet.
            container_out (av.container.output.OutputContainer): Output container.

        Returns:
            None: Processes and muxes the audio packet.
        """
        if self.stream_out is None:
            return

        # Copy mode remuxes packets directly. Re-encode mode must decode,
        # optionally resample, and then encode new packets.
        if self.copy_mode:
            try:
                packet.stream = self.stream_out
                container_out.mux(packet)
            except Exception:
                pass
            return

        try:
            for input_frame in packet.decode():
                if not isinstance(input_frame, av.AudioFrame):
                    continue

                if self.resampler is None:
                    output_frames = [input_frame]
                else:
                    # PyAV resampling may yield one frame, many frames, or None,
                    # so normalize the result before encoding.
                    raw = self.resampler.resample(input_frame)
                    output_frames = (
                        list(raw) if isinstance(raw, (list, tuple)) else [raw]
                    )

                for frame in output_frames:
                    if frame is None:
                        continue
                    for output_packet in self.stream_out.encode(frame):
                        container_out.mux(output_packet)
        except Exception:
            pass

    def flush(self, container_out):
        """
        Flush delayed audio packets into the output container.

        Args:
            container_out (av.container.output.OutputContainer): Output container.

        Returns:
            None: Flushes any pending audio packets to the output container.
        """
        if self.stream_out is None or self.copy_mode:
            return
        try:
            for output_packet in self.stream_out.encode(None):
                container_out.mux(output_packet)
        except Exception:
            pass


def require_not_cancelled(check_cancel):
    """
    Raise when cancellation was requested by the job manager.

    Args:
        check_cancel (callable | None): Optional callable that returns True when work should stop.

    Returns:
        None: Raises JobCancelled if cancellation was requested.
    """
    if check_cancel and check_cancel():
        raise JobCancelled()


def update_frame_progress(progress, progress_cb, processed, total):
    """
    Send a frame-progress update.

    Args:
        progress (StreamlitProgress | None): Optional foreground progress helper.
        progress_cb (callable | None): Optional background callback.
        processed (int): Number of processed frames.
        total (int): Estimated total frame count.

    Returns:
        None: Updates the progress callback or foreground widgets.
    """
    if progress_cb is not None:
        progress_cb(processed, total)
        return
    if progress is not None:
        progress.update(processed, total)


def build_video_encoding_options(codec, encoding_mode, crf_value, preset):
    """
    Build minimal encoder options for the chosen codec.

    Args:
        codec (str): Output codec name.
        encoding_mode (str): ``CRF``, ``VBR``, or ``CBR``.
        crf_value (int): CRF value from the UI.
        preset (str): Encoder preset.

    Returns:
        dict: Encoder options dictionary.
    """
    options = {"preset": preset}
    if encoding_mode != "CRF":
        return options

    if codec not in ("libx264", "libx265", "libvpx-vp9"):
        return options

    options["crf"] = str(int(crf_value))
    if codec == "libvpx-vp9":
        # VP9 gets a couple codec-specific defaults here so CRF mode stays
        # usable without exposing extra encoder knobs in the UI.
        options["row-mt"] = "1"
        options["b"] = "0"
    return options


def add_video_output_stream(container_out, video_stream_in, target_w, target_h, codec):
    """
    Create the output video stream.

    Args:
        container_out (av.container.output.OutputContainer): Output container.
        video_stream_in (object): Input video stream.
        target_w (int): Output width.
        target_h (int): Output height.
        codec (str): Output codec name.

    Returns:
        object: Configured output video stream.
    """
    stream_out_v = container_out.add_stream(codec, rate=video_stream_in.average_rate)
    stream_out_v.width = int(target_w)
    stream_out_v.height = int(target_h)
    stream_out_v.pix_fmt = "yuv420p"
    return stream_out_v


def configure_video_output_stream(
    stream_out_v,
    codec,
    encoding_mode,
    crf_value,
    bitrate_kbps,
    preset,
):
    """
    Apply codec-specific settings to the output stream.

    Args:
        stream_out_v (object): Output video stream.
        codec (str): Output codec name.
        encoding_mode (str): ``CRF``, ``VBR``, or ``CBR``.
        crf_value (int): CRF value from the UI.
        bitrate_kbps (int | None): Optional bitrate in kbps.
        preset (str): Encoder preset.

    Returns:
        str | None: Error text, or None when configuration succeeds.
    """
    options = build_video_encoding_options(codec, encoding_mode, crf_value, preset)

    if encoding_mode in ("VBR", "CBR"):
        if bitrate_kbps is None:
            return "Bitrate must be set for VBR/CBR."
        target_bps = int(bitrate_kbps) * 1000
        stream_out_v.bit_rate = target_bps
        if encoding_mode == "VBR":
            options["maxrate"] = f"{target_bps * 2}k"
            options["bufsize"] = f"{target_bps * 2}k"

    if encoding_mode == "CRF" and codec == "libvpx-vp9":
        stream_out_v.bit_rate = 0

    stream_out_v.options = options
    return None
