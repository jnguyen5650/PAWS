import av
import numpy as np
import io
import warnings


def degrade_clip_with_avc(frames: np.ndarray, crf=40, gop=10, preset='veryfast', tune=None, bitrate=None):
    """
    Simulates H.264 (AVC) compression artifacts on a sequence of video frames via an in-memory
    encode-decode cycle using the libx264 encoder. Frames are padded to ensure even height and width,
    as required by YUV420p format, with edge-value replication. After decoding, frames are cropped
    back to the original dimensions.

    Arguments:
        frames (np.ndarray): Input array of shape (T, H, W, 3), dtype uint8, with channels in RGB order.
        crf (int): Constant Rate Factor. Lower values preserve more detail, higher values induce greater loss.
        gop (int): Distance between keyframes (Group Of Pictures).
        bitrate (int or None): If specified, sets target bitrate in bits per second, overriding CRF.

    Returns:
        np.ndarray: Array of decoded frames, shape (T, H, W, C), dtype uint8.
    """
    T, H, W, C = frames.shape

    pad_h = (H % 2)
    pad_w = (W % 2)

    if pad_h or pad_w:
        padded = []
        for f in frames:
            pad_config = ((0, pad_h), (0, pad_w), (0, 0))
            padded.append(np.pad(f, pad_config, mode='edge'))
        frames_enc = np.stack(padded, axis=0)
        H_enc, W_enc = H + pad_h, W + pad_w
    else:
        frames_enc = frames
        H_enc, W_enc = H, W

    buffer = io.BytesIO()
    output = av.open(buffer, mode='w', format='mp4')
    stream = output.add_stream('libx264', rate=30)
    stream.width  = W_enc
    stream.height = H_enc
    stream.pix_fmt = 'yuv420p'
    opts = {
        'crf': str(crf),
        'preset': preset,
        'g': str(gop),
        'keyint_min': str(gop // 2),
    }
    if tune:
        opts['tune'] = tune
    stream.options = opts

    if bitrate:
        stream.bit_rate = bitrate
        del opts['crf']

    for frame in frames_enc:
        vf = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(vf):
            output.mux(packet)
    for packet in stream.encode(None):
        output.mux(packet)
    output.close()

    buffer.seek(0)
    container = av.open(buffer, mode='r')
    vstream = next(s for s in container.streams if s.type == 'video')
    decoded = []
    for packet in container.demux(vstream):
        for frame in packet.decode():
            decoded.append(frame.to_ndarray(format='rgb24'))
    container.close()

    decoded = np.stack(decoded, axis=0)

    if pad_h or pad_w:
        decoded = decoded[:, :H, :W, :]

    return decoded
