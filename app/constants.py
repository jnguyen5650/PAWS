"""
Define shared constants for the PAWS Streamlit app.
"""

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent

APP_TITLE = "\U0001f43e PAWS: Practical Video Super-Resolution"
APP_SUBTITLE = "Open-source batch inference for published PAWS checkpoints."

DEFAULT_OUT_DIR = str(Path.home() / "Downloads" / "PAWS_Upscaled_Videos")
DEFAULT_MODEL_DIR = str(REPO_ROOT / "outputs" / "published_models")

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
PUBLISHED_MODEL_EXT = ".paws.pth"
PAWS_MODEL_FORMAT = "paws_published_model"
PAWS_MODEL_VERSION = 1

DEFAULT_TEMPORAL_CHUNK = 30
DEFAULT_TILE_HEIGHT = 128
DEFAULT_TILE_WIDTH = 128
MIN_TEMPORAL_TILE = 2
MIN_SPATIAL_TILE = 64
DEFAULT_PAD_H = 20
DEFAULT_PAD_W = 20

PRECISION_OPTIONS = ("fp32", "bf16", "fp16")
COMPILE_MODE_OPTIONS = ("none", "default", "reduce-overhead", "max-autotune")

CONTAINER_EXT_MAP = {
    "MP4 (H.264, H.265)": ".mp4",
    "WEBM (VP9)": ".webm",
    "MKV (Matroska)": ".mkv",
}

MIME_MAP = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
}

CONTAINER_TO_CODECS_UI = {
    "MP4 (H.264, H.265)": [
        "libx264 (H.264/AVC)",
        "libx265 (H.265/HEVC)",
    ],
    "WEBM (VP9)": [
        "libvpx-vp9 (VP9)",
    ],
    "MKV (Matroska)": [
        "libx264 (H.264/AVC)",
        "libx265 (H.265/HEVC)",
        "libvpx-vp9 (VP9)",
    ],
}

ICON_FOLDER = "\U0001f4c1"
ICON_SPARKLES = "\u2728"
ICON_TOOLS = "\U0001f6e0\ufe0f"
ICON_INFO = "\u2139\ufe0f"
ICON_WARNING = "\u26a0\ufe0f"
ICON_ERROR = "\u274c"
ICON_CHECK = "\u2705"
ICON_BROOM = "\U0001f9f9"
ICON_THREADS = "\U0001f9f5"
ICON_STOP = "\u26d4"
ICON_SAVE = "\U0001f4be"
ICON_CREATE = "\U0001f6e0\ufe0f"
ICON_ROCKET = "\U0001f680"
