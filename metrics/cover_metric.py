import os
import random
import re
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from cover.datasets import UnifiedFrameSampler, get_single_view
from cover.models import COVER


MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])

MEAN_CLIP = torch.FloatTensor([122.77, 116.75, 104.09])
STD_CLIP = torch.FloatTensor([68.50, 66.63, 70.32])

COVER_COMMIT = "7373bee6bc55fdc2b617e40c8c45c8295c5e6467"


def _natural_key(s):
    """
    Build a natural-sort key for filenames.

    Args:
        s (str): Filename string.

    Returns:
        list: Key used for natural sorting so frame_10 comes after frame_2.
    """
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", s)
    ]


def _download_file(url, dst_path, timeout=300):
    """
    Download a URL to a local file.

    Args:
        url (str): Remote URL.
        dst_path (str): Local path to write.
        timeout (int): Network timeout in seconds.

    Returns:
        str: The destination path string.

    Raises:
        URLError: If the download request fails.
        OSError: If the destination file cannot be written or replaced.
    """
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    tmp_path = dst_path + ".tmp"

    with urllib.request.urlopen(url, timeout=timeout) as r:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(r, f)

    os.replace(tmp_path, dst_path)
    return dst_path


def _ensure_cover_assets(assets_dir):
    """
    Ensure cover.yml and COVER.pth exist and are valid.

    Args:
        assets_dir (str or Path): Directory where assets should be stored.

    Returns:
        tuple: Paths to cover.yml and COVER.pth as strings.

    Raises:
        OSError: If asset directories or files cannot be created or replaced.
    """
    assets_dir = Path(assets_dir)
    weights_dir = assets_dir / "pretrained_weights"
    assets_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    cover_yml_path = assets_dir / "cover.yml"
    cover_pth_path = weights_dir / "COVER.pth"

    cover_yml_url = (
        "https://raw.githubusercontent.com/taco-group/COVER/"
        + COVER_COMMIT
        + "/cover.yml"
    )
    cover_pth_url = (
        "https://github.com/taco-group/COVER/raw/refs/heads/"
        "release/Model/COVER.pth"
    )

    if not cover_yml_path.exists():
        _download_file(cover_yml_url, str(cover_yml_path))

    if not cover_pth_path.exists():
        _download_file(cover_pth_url, str(cover_pth_path))

    return str(cover_yml_path), str(cover_pth_path)


def _list_frame_paths(frames_dir):
    """
    List frame image files in natural filename order.

    Args:
        frames_dir (str or Path): Directory containing image frames.

    Returns:
        list: Path objects for image files.

    Raises:
        FileNotFoundError: If frames_dir is not a directory.
        ValueError: If frames_dir contains no supported image files.
    """
    p = Path(frames_dir)
    if not p.is_dir():
        raise FileNotFoundError("Sequence directory not found: {}".format(p))

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = [
        x for x in p.iterdir()
        if x.is_file() and x.suffix.lower() in exts
    ]
    paths.sort(key=lambda x: _natural_key(x.name))

    if not paths:
        raise ValueError("No image frames found in {}".format(p))

    return paths


def _read_frame_rgb_tensor(path):
    """
    Read one image and return an RGB uint8 torch tensor in HWC layout.

    Args:
        path (str or Path): Path to an image file.

    Returns:
        torch.Tensor: RGB tensor with shape (H, W, 3), dtype uint8.

    Raises:
        ValueError: If the image cannot be read or has an unsupported shape.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to read image: {}".format(path))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] >= 3:
        img = img[:, :, :3]
    else:
        raise ValueError(
            "Unsupported image shape for {}: {}".format(path, img.shape)
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img)


def _build_temporal_samplers(sample_types):
    """
    Build COVER temporal samplers from config sample_types.

    Args:
        sample_types (dict): The sample_types block from cover.yml.

    Returns:
        dict: Branch name to UnifiedFrameSampler.
    """
    samplers = {}
    for stype, sopt in sample_types.items():
        samplers[stype] = UnifiedFrameSampler(
            sopt["clip_len"] // sopt["t_frag"],
            sopt["t_frag"],
            sopt["frame_interval"],
            sopt["num_clips"],
        )
    return samplers


def _outputs_to_cover_score(outputs):
    """
    Convert COVER model outputs into final overall score.

    Args:
        outputs (list): COVER outputs for semantic, technical, and aesthetic
            branches.

    Returns:
        float: Overall score equal to semantic + technical + aesthetic.
    """
    vals = [float(x.mean().item()) for x in outputs]
    return float(vals[0] + vals[1] + vals[2])


class CoverMetric:
    """
    COVER metric wrapper with deterministic fixed-seed averaging.

    The default configuration scores each sequence with seeds [0, 1, 2, 3, 4]
    and returns the mean COVER score. COVER assets are stored under
    metrics/cover_assets unless assets_dir is provided; missing assets are
    downloaded when auto_download_assets is True.

    Returns:
        CoverMetric: Callable object that scores one sequence folder.
    """

    def __init__(
        self,
        device,
        assets_dir=None,
        dataset_key="val-ytugc",
        auto_download_assets=True,
        num_samples=5,
        base_seed=0,
    ):
        """
        Initialize the COVER model and config.

        Args:
            device (str or torch.device): Device used to run COVER.
            assets_dir (str or Path, optional): Directory containing cover.yml
                and pretrained_weights/COVER.pth.
            dataset_key (str): Config dataset key in cover.yml.
            auto_download_assets (bool): Whether to download missing assets.
            num_samples (int): Number of fixed stochastic COVER samples to
                average.
            base_seed (int): First seed in the fixed global seed sequence.

        Returns:
            None.

        Raises:
            FileNotFoundError: If required COVER assets are missing.
        """
        self.device_t = torch.device(str(device))
        self.dataset_key = dataset_key
        self.num_samples = num_samples
        self.base_seed = base_seed

        if assets_dir is None:
            assets_dir = Path(__file__).resolve().parent / "cover_assets"
        else:
            assets_dir = Path(assets_dir)

        if auto_download_assets:
            self.cover_yml_path, self.cover_pth_path = _ensure_cover_assets(
                assets_dir
            )
        else:
            self.cover_yml_path = str(assets_dir / "cover.yml")
            self.cover_pth_path = str(
                assets_dir / "pretrained_weights" / "COVER.pth"
            )

        if not os.path.exists(self.cover_yml_path):
            raise FileNotFoundError(
                "Missing cover.yml: {}".format(self.cover_yml_path)
            )
        if not os.path.exists(self.cover_pth_path):
            raise FileNotFoundError(
                "Missing COVER.pth: {}".format(self.cover_pth_path)
            )

        with open(self.cover_yml_path, "r", encoding="utf-8") as f:
            self.opt = yaml.safe_load(f)

        self.sample_types = self.opt["data"][self.dataset_key]["args"][
            "sample_types"
        ]
        self.temporal_samplers = _build_temporal_samplers(self.sample_types)

        self.evaluator = COVER(**self.opt["model"]["args"]).to(self.device_t)

        # COVER.pth stores a full checkpoint, so force old-style loading.
        ckpt = torch.load(
            self.cover_pth_path,
            map_location=self.device_t,
            weights_only=False,
        )

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        self.evaluator.load_state_dict(state_dict, strict=False)
        self.evaluator.eval()

    def _score_once(self, frame_paths, seed):
        """
        Compute one seeded COVER sample for a sequence.

        This runs temporal sampling, COVER view construction, model inference,
        and output conversion once under the provided seed.

        Args:
            frame_paths (list): Ordered frame paths for one sequence.
            seed (int): Seed used for Python, NumPy, and Torch samplers.

        Returns:
            float: COVER score for one sampled view set.

        Raises:
            ValueError: If a frame cannot be read or has an unsupported shape.
            KeyError: If cover.yml contains an unexpected COVER branch.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        num_frames = len(frame_paths)

        frame_inds = {}
        all_inds = []

        for stype, sampler in self.temporal_samplers.items():
            inds = np.array(sampler(num_frames), dtype=np.int32)
            inds = np.mod(inds, num_frames)
            frame_inds[stype] = inds
            all_inds.append(inds)

        unique_inds = np.unique(np.concatenate(all_inds, axis=0))
        frame_cache = {}
        for idx in unique_inds:
            frame_cache[int(idx)] = _read_frame_rgb_tensor(
                frame_paths[int(idx)]
            )

        raw_views = {}
        for stype in self.sample_types.keys():
            imgs = [frame_cache[int(i)] for i in frame_inds[stype]]
            # [T,H,W,C] -> [C,T,H,W]
            video = torch.stack(imgs, dim=0).permute(3, 0, 1, 2)
            raw_views[stype] = get_single_view(
                video,
                stype,
                **self.sample_types[stype],
            )

        views = {}
        for k, v in raw_views.items():
            num_clips = self.sample_types[k].get("num_clips", 1)

            if k == "technical" or k == "aesthetic":
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - MEAN) / STD)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, v.shape[2], v.shape[3])
                    .transpose(0, 1)
                    .to(self.device_t)
                )
            elif k == "semantic":
                views[k] = (
                    ((v.permute(1, 2, 3, 0) - MEAN_CLIP) / STD_CLIP)
                    .permute(3, 0, 1, 2)
                    .reshape(v.shape[0], num_clips, -1, v.shape[2], v.shape[3])
                    .transpose(0, 1)
                    .to(self.device_t)
                )
            else:
                raise KeyError("Unexpected COVER branch: {}".format(k))

        outputs = self.evaluator(views)
        return _outputs_to_cover_score(outputs)

    @torch.inference_mode()
    def __call__(self, frames_dir):
        """
        Score one sequence directory with averaged fixed-seed COVER samples.

        The original RNG states are restored after scoring so deterministic
        COVER sampling does not affect callers.

        Args:
            frames_dir (str or Path): Directory containing ordered frame
                images.

        Returns:
            float: Mean COVER score over the fixed seed sequence.

        Raises:
            FileNotFoundError: If frames_dir is not a directory.
            ValueError: If frames_dir has no supported image files or a frame
                cannot be read.
            KeyError: If cover.yml contains an unexpected COVER branch.
        """
        frame_paths = _list_frame_paths(frames_dir)

        random_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()

        try:
            sample_scores = []
            for sample_index in range(self.num_samples):
                seed = self.base_seed + sample_index
                sample_scores.append(self._score_once(frame_paths, seed))

            return float(np.mean(sample_scores))
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
