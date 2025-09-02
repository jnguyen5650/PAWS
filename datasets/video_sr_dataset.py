import os
import inspect
import random
import math
import numpy as np
import warnings
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import io as tv_io
from .degradations import degrade_clip_with_avc
from .degradation_kernels import filter2D, circular_lowpass_kernel, random_mixed_kernels


class VideoSRDataset(Dataset):
    def __init__(
        self,
        lr_dir: str,
        hr_dir: str,
        scale_factor: int = 4,
        patch_size: int = 64,
        num_frames: int = 14,
        use_degradations: bool = False,
        real_opt: dict = None,
        use_custom_degradations: bool = False,
        custom_opt: dict = None,
    ):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.num_frames = num_frames

        self.use_degradations = use_degradations
        self.degradation_opts = real_opt or {}

        self.use_custom_degradations = use_custom_degradations
        custom_opt = custom_opt or {}
        self.custom_final_resize_modes = custom_opt.get('final_resize_modes', ['area','bilinear','bicubic'])
        self.custom_degradation_pipeline = custom_opt.get('pipeline', [])

        self.DEGRADATION_REGISTRY = {
            "sinc": self._degrade_sinc,
            "blur": self._degrade_blur,
            "resize": self._degrade_resize,
            "gaussian_noise": self._degrade_gaussian_noise,
            "poisson_noise": self._degrade_poisson_noise,
            "jpeg_compress": self._jpeg_compress,
            "avc_compress": self._avc_compress,
        }

        if use_degradations and use_custom_degradations:
            raise ValueError("Cannot use both Real-ESRGAN and custom degradation pipelines simultaneously.")

        def _valid_odd(k: int) -> int:
            original = k
            k = int(k)
            if k < 7:
                warnings.warn(
                    f"Provided blur/sinc kernel size={original} is < 7. "
                    f"Using 7 (minimum supported).",
                    RuntimeWarning
                )
                k = 7
            if (k & 1) == 0:  # even -> bump to next odd
                warnings.warn(
                    f"Provided blur/sinc kernel size={original} is even. "
                    f"Using {k+1} (next odd).",
                    RuntimeWarning
                )
                k += 1
            return k

        # Real-ESRGAN pipeline configs
        cfg = self.degradation_opts

        # Real-ESRGAN Stage 1 configs
        self.blur_kernel_size = _valid_odd(cfg.get('blur_kernel_size', 21))
        self.kernel_list_stage1 = cfg.get('kernel_list', [])
        self.kernel_probs_stage1 = cfg.get('kernel_prob', [])
        self.blur_sigma_range1 = cfg.get('blur_sigma_range', [0.2,3.0])
        self.betag_range1 = cfg.get('betag_range', [0.5,4.0])
        self.betap_range1 = cfg.get('betap_range', [1.0,2.0])
        self.sinc_prob1 = cfg.get('sinc_prob', 0.1)
        # Real-ESRGAN Stage 2 configs
        self.blur_kernel_size2 = _valid_odd(cfg.get('blur_kernel_size2', 21))
        self.second_blur_prob = cfg.get('second_blur_prob', 0.8)
        self.kernel_list_stage2 = cfg.get('kernel_list2', [])
        self.kernel_probs_stage2 = cfg.get('kernel_prob2', [])
        self.blur_sigma_range2 = cfg.get('blur_sigma_range2', [0.2,1.5])
        self.betag_range2 = cfg.get('betag_range2', [0.5,4.0])
        self.betap_range2 = cfg.get('betap_range2', [1.0,2.0])
        self.sinc_prob2 = cfg.get('sinc_prob2', 0.1)
        self.final_sinc_prob = cfg.get('final_sinc_prob', 0.8)
        # AVC settings
        self.avc_crf_range = cfg.get("avc_crf_range", (10, 37))
        self.avc_gop = cfg.get("avc_gop", 60)
        self.avc_preset = cfg.get("avc_preset", 'veryfast')
        self.avc_tune = cfg.get("avc_tune", None)
        self.avc_passes = cfg.get("avc_passes", 2)
        self.avc_prob = cfg.get("avc_prob", 0.8)

        # Precompute possible kernel sizes and an impulse kernel for bypass
        self.available_kernel_sizes = list(range(7, self.blur_kernel_size + 1, 2))
        self.available_kernel_sizes2 = list(range(7, self.blur_kernel_size2 + 1, 2))

        # Build the global max size including custom steps
        max_size = max(self.blur_kernel_size, self.blur_kernel_size2)
        if self.use_custom_degradations:
            for step in self.custom_degradation_pipeline:
                if step.get('type') in ('blur','sinc'):
                    for s in step.get('kernel_sizes', []):
                        max_size = max(max_size, _valid_odd(s))
        impulse = torch.zeros((1, max_size, max_size), dtype=torch.float32)
        impulse[0, max_size // 2, max_size // 2] = 1.0
        self.impulse_kernel = impulse
        assert (self.impulse_kernel.size(1) % 2) == 1, "Impulse kernel must be odd."

        # Aggregate all scale factors from resizes (built-in and custom)
        max_scale = 1.0
        if self.use_degradations:
            max_scale *= self.degradation_opts['resize_range'][1]
            max_scale *= self.degradation_opts['resize_range2'][1]
        if self.use_custom_degradations:
            for step in self.custom_degradation_pipeline:
                if step['type'] == 'resize':
                    max_scale *= step.get('resize_range', [1.0, 1.0])[1]

        # Margin is radius * total upsample, for full context throughout pipeline
        self.margin   = math.ceil((max_size // 2) * max_scale)
        self.big_patch = self.patch_size + 2 * self.margin

        # Scan video directories for paired LR/HR frame lists
        self.video_dirs = sorted(
            d for d in os.listdir(lr_dir)
            if os.path.isdir(os.path.join(lr_dir, d))
        )
        self.video_list = []
        for video_id in self.video_dirs:
            lr_path = os.path.join(lr_dir, video_id)
            hr_path = os.path.join(hr_dir, video_id)
            if not os.path.isdir(hr_path):
                continue
            lr_frames = sorted(
                f for f in os.listdir(lr_path)
                if f.lower().endswith(('.png','.jpg','.jpeg'))
            )
            hr_frames = sorted(
                f for f in os.listdir(hr_path)
                if f.lower().endswith(('.png','.jpg','.jpeg'))
            )
            if lr_frames and hr_frames:
                self.video_list.append({
                    'video_id': video_id,
                    'lr_frame_names': lr_frames,
                    'hr_frame_names': hr_frames
                })

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index: int):
        video_info = self.video_list[index]
        lr_video_path = os.path.join(self.lr_dir, video_info['video_id'])
        hr_video_path = os.path.join(self.hr_dir, video_info['video_id'])

        # Temporal sampling: select contiguous frame window of length num_frames
        total_frames = len(video_info['lr_frame_names'])
        if total_frames >= self.num_frames:
            start_idx = random.randint(0, total_frames - self.num_frames)
            sel_lr = video_info['lr_frame_names'][start_idx:start_idx + self.num_frames]
            sel_hr = video_info['hr_frame_names'][start_idx:start_idx + self.num_frames]
        else:
            sel_lr = video_info['lr_frame_names']
            sel_hr = video_info['hr_frame_names']

        # Read spatial dims of first LR frame (for cropping)
        first_lr = tv_io.read_image(os.path.join(lr_video_path, sel_lr[0])).float() / 255.0
        _, H_lr, W_lr = first_lr.shape
        if H_lr < self.big_patch or W_lr < self.big_patch:
            raise ValueError(
                f"Need at least {self.big_patch}x{self.big_patch} context, got {H_lr}x{W_lr}"
            )

        # Random top-left for context window
        x0 = random.randint(0, W_lr - self.big_patch)
        y0 = random.randint(0, H_lr - self.big_patch)

        # Load and crop big_patch window for all frames
        hr_big, lr_big = [], []
        for lr_name, hr_name in zip(sel_lr, sel_hr):
            lr_img = tv_io.read_image(os.path.join(lr_video_path, lr_name)).float() / 255.0
            hr_img = tv_io.read_image(os.path.join(hr_video_path, hr_name)).float() / 255.0

            _, H_hr, W_hr = hr_img.shape
            expected_hr_h = H_lr * self.scale_factor
            expected_hr_w = W_lr * self.scale_factor
            if H_hr != expected_hr_h or W_hr != expected_hr_w:
                raise ValueError(
                    f"HR frame dimensions ({H_hr}x{W_hr}) do not match expected "
                    f"({expected_hr_h}x{expected_hr_w}) for video '{video_info['video_id']}', "
                    f"frame '{hr_name}'. Check your scale_factor and input sizes."
                )

            # LR crop
            lr_crop = lr_img[:, y0:y0 + self.big_patch, x0:x0 + self.big_patch]

            # HR crop (aligned by scale_factor)
            y0_hr = y0 * self.scale_factor
            x0_hr = x0 * self.scale_factor
            big_hr = self.big_patch * self.scale_factor
            hr_crop = hr_img[:, y0_hr:y0_hr + big_hr, x0_hr:x0_hr + big_hr]

            lr_big.append(lr_crop)
            hr_big.append(hr_crop)

        hr_big_clip = torch.stack(hr_big, dim=0)  # (T, C, big_hr, big_hr)

        # Apply degradation pipeline or use direct LR crop
        if self.use_degradations:
            lq_big_clip = self._run_degradation(hr_big_clip)
        elif self.use_custom_degradations:
            lq_big_clip = self._run_custom_degradation(hr_big_clip)
        else:
            lq_big_clip = torch.stack(lr_big, dim=0)

        # Extract central patch_size region: removes all context/margin
        cy, cx = self.margin, self.margin
        ps     = self.patch_size
        # Low-quality sub-patch
        lq_patches = lq_big_clip[:, :, cy:cy + ps, cx:cx + ps]
        # High-quality sub-patch
        hr_patches = hr_big_clip[:, :,
                    cy * self.scale_factor:(cy + ps) * self.scale_factor,
                    cx * self.scale_factor:(cx + ps) * self.scale_factor
                    ]

        # Random horizontal flip for data augmentation
        if random.random() < 0.5:
            lq_patches = torch.flip(lq_patches, dims=[3])
            hr_patches = torch.flip(hr_patches, dims=[3])

        return lq_patches, hr_patches


    def _run_degradation(self, hr_clip: torch.Tensor) -> torch.Tensor:
        device = hr_clip.device
        scale = self.scale_factor
        _, _, H_hr, W_hr = hr_clip.shape
        H_lq, W_lq = H_hr // scale, W_hr // scale

        # ------ Stage 1: kernel sampling & blur ------
        if random.random() < self.sinc_prob1:
            out = self._degrade_sinc(hr_clip, self.available_kernel_sizes)
        else:
            out = self._degrade_blur(
                hr_clip,
                self.kernel_list_stage1,
                self.kernel_probs_stage1,
                self.available_kernel_sizes,
                self.blur_sigma_range1,
                self.betag_range1,
                self.betap_range1,
                noise_range=None
            )

        # ------ Stage 1: resize & noise ------
        out = self._degrade_resize(
            out,
            self.degradation_opts['resize_prob'],
            self.degradation_opts['resize_range'],
            modes=self.degradation_opts['resize_modes']
        )
        if random.random() < self.degradation_opts['gaussian_noise_prob']:
            out = self._degrade_gaussian_noise(
                out,
                self.degradation_opts['noise_range'],
                self.degradation_opts.get('gray_noise_prob', 0)
            )
        else:
            out = self._degrade_poisson_noise(
                out,
                self.degradation_opts['poisson_scale_range']
            )

        # ------ Stage 1: compression ------
        if random.random() < self.avc_prob:
            out = self._avc_compress(out, self.avc_crf_range, self.avc_gop, self.avc_preset, self.avc_tune)
        else:
            out = self._jpeg_compress(out, self.degradation_opts['jpeg_range'])

        # ------ Stage 2: kernel sampling & blur ------  
        if random.random() < self.second_blur_prob:
            if random.random() < self.sinc_prob2:
                out = self._degrade_sinc(out, self.available_kernel_sizes2)
            else:
                out = self._degrade_blur(
                    out,
                    self.kernel_list_stage2,
                    self.kernel_probs_stage2,
                    self.available_kernel_sizes2,
                    self.blur_sigma_range2,
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None
                )

        # ------ Stage 2: resize & noise ------
        out = self._degrade_resize(
            out,
            self.degradation_opts['resize_prob2'],
            self.degradation_opts['resize_range2'],
            modes=self.degradation_opts['resize_modes2']
        )
        if random.random() < self.degradation_opts['gaussian_noise_prob2']:
            out = self._degrade_gaussian_noise(
                out,
                self.degradation_opts['noise_range2'],
                self.degradation_opts.get('gray_noise_prob2', 0)
            )
        else:
            out = self._degrade_poisson_noise(
                out,
                self.degradation_opts['poisson_scale_range2']
            )

        # ------ Final stage: optional sinc ------
        if random.random() < self.final_sinc_prob:
            kf_size = random.choice(self.available_kernel_sizes2)
            cutoff = random.uniform(math.pi/3, math.pi)
            pad_to = int(self.impulse_kernel.size(1))
            final_kernel = circular_lowpass_kernel(cutoff, kf_size, pad_to=pad_to)
            final_kernel = torch.from_numpy(final_kernel).to(device)
        else:
            final_kernel = self.impulse_kernel.to(device)

        # Two possible orders: resize -> filter -> compress or compress -> resize -> filter
        if random.random() < 0.5:
            out = F.interpolate(out, size=(H_lq, W_lq), mode=random.choice(self.degradation_opts.get('resize_modes2', ['area','bilinear','bicubic'])))
            out = filter2D(out, final_kernel).clamp(0.0, 1.0)
            if random.random() < self.avc_prob and self.avc_passes > 1:
                out = self._avc_compress(out, self.avc_crf_range, self.avc_gop, self.avc_preset, self.avc_tune)
            else:
                out = self._jpeg_compress(out, self.degradation_opts['jpeg_range2'])
        else:
            if random.random() < self.avc_prob and self.avc_passes > 1:
                out = self._avc_compress(out, self.avc_crf_range, self.avc_gop, self.avc_preset, self.avc_tune)
            else:
                out = self._jpeg_compress(out, self.degradation_opts['jpeg_range2'])
            out = F.interpolate(out, size=(H_lq, W_lq), mode=random.choice(self.degradation_opts.get('resize_modes2', ['area','bilinear','bicubic'])))
            out = filter2D(out, final_kernel).clamp(0.0, 1.0)

        return out.clamp(0.0, 1.0)


    def _run_custom_degradation(self, hr_clip: torch.Tensor) -> torch.Tensor:
        _, _, H_hr, W_hr = hr_clip.shape
        H_lq, W_lq = H_hr // self.scale_factor, W_hr // self.scale_factor

        out = hr_clip
        for step in self.custom_degradation_pipeline:
            prob = step.get("prob", 1.0)
            # If prob is a list, pass inside function (usually for resize);
            # skip only if scalar and fail prob check
            if not isinstance(prob, list) and random.random() > prob:
                continue

            step_type = step['type']
            func = self.DEGRADATION_REGISTRY[step_type]
            func_args = inspect.getfullargspec(func).args
            if func_args[0] == 'self':
                func_args = func_args[1:]
            kwargs = {k: v for k, v in step.items() if k in func_args}
            out = func(out, **kwargs)
        out = F.interpolate(out, size=(H_lq, W_lq), mode=random.choice(self.custom_final_resize_modes))
        return out.clamp(0.0, 1.0)


    def _degrade_sinc(
        self,
        clip: torch.Tensor,
        kernel_sizes: list[int],
    ) -> torch.Tensor:
        device = clip.device

        k_size = random.choice(kernel_sizes)
        
        cutoff = (random.uniform(math.pi/3, math.pi)
                  if k_size < 13 else random.uniform(math.pi/5, math.pi))
        kern = circular_lowpass_kernel(cutoff, k_size, pad_to=0)
        kern_t = torch.from_numpy(kern).to(device)
        return filter2D(clip, kern_t).clamp(0.0, 1.0)


    def _degrade_blur(
        self,
        clip: torch.Tensor,
        kernel_list,
        kernel_prob,
        kernel_sizes,
        sigma_range,
        betag_range,
        betap_range,
        noise_range=None
    ) -> torch.Tensor:
        device = clip.device

        k_size = random.choice(kernel_sizes)

        kern = random_mixed_kernels(
                kernel_list, kernel_prob,
                k_size, sigma_range, sigma_range,
                [-math.pi, math.pi],
                betag_range, betap_range,
                noise_range=noise_range
            )  

        pad = (self.impulse_kernel.size(1) - k_size) // 2
        kern = np.pad(kern, ((pad,pad),(pad,pad)))
        kern = kern.astype(np.float32)
        kern /= kern.sum()

        kernel_t = torch.from_numpy(kern).to(device)
        return filter2D(clip, kernel_t).clamp(0.0, 1.0)
    
    
    def _degrade_resize(
        self,
        clip: torch.Tensor,
        resize_prob,
        resize_range,
        modes=('area','bilinear','bicubic')
    ) -> torch.Tensor:
        choice = random.choices(['up','down','keep'], resize_prob)[0]
        scale_factor = {'up': random.uniform(1, resize_range[1]), 'down': random.uniform(resize_range[0], 1), 'keep':1}[choice]
        out = F.interpolate(clip, scale_factor=scale_factor, mode=random.choice(modes))
        return out
    

    def _degrade_gaussian_noise(
        self,
        clip: torch.Tensor,
        sigma_range,
        gray_noise_prob
    ) -> torch.Tensor:
        sigma = random.uniform(*sigma_range) / 255.0

        if random.random() < gray_noise_prob:
            b, c, h, w = clip.shape
            noise_gray = torch.randn(b, 1, h, w, device=clip.device) * sigma
            noise = noise_gray.expand(b, c, h, w)
        else:
            noise = torch.randn_like(clip) * sigma

        return (clip + noise).clamp(0.0, 1.0)
    

    def _degrade_poisson_noise(
        self,
        clip: torch.Tensor,
        scale_p_range: tuple
    ) -> torch.Tensor:
        # Poisson shot noise (Real–ESRGAN style)
        scale_p = random.uniform(*scale_p_range)
        img_q = torch.clamp((clip * 255.0).round(), 0, 255) / 255.0
        vals = img_q.new_full((img_q.size(0), 1, 1, 1), 256.0)
        sampled = torch.poisson(img_q * vals) / vals
        shot_noise = sampled - img_q

        return (img_q + shot_noise * scale_p).clamp(0.0, 1.0)


    def _jpeg_compress(self, clip: torch.Tensor, quality_range) -> torch.Tensor:
        compressed_frames = []
        for frame in clip:
            img_u8 = (frame * 255.0).to(torch.uint8)
            q = random.randint(*quality_range)
            enc = tv_io.encode_jpeg(img_u8, quality=q)
            dec = tv_io.decode_jpeg(enc).float() / 255.0
            compressed_frames.append(dec)
        return torch.stack(compressed_frames, dim=0)


    def _avc_compress(self, clip: torch.Tensor, crf_range, gop, preset, tune) -> torch.Tensor:
        arr = (clip.permute(0,2,3,1).cpu().numpy() * 255).astype(np.uint8)
        crf = random.randint(*crf_range)
        dec = degrade_clip_with_avc(arr, crf=crf, gop=gop, preset=preset, tune=tune)
        return torch.from_numpy(dec).permute(0,3,1,2).float() / 255.0
