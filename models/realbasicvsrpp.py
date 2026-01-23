import torch
import torch.nn as nn
import inspect
from typing import Any, Dict, Optional

from .basicvsrpp import BasicVSRPlusPlus
from .archs.arch_util import ResidualBlockNoBN
from .archs.rrdb_arch import CleaningModuleRRDB
from .archs.swinir_arch import CleaningModuleRSTB
from .archs.hat_arch import CleaningModuleHAT
from .registry import register_model


@register_model()
class RealBasicVSRPlusPlus(nn.Module):
    def __init__(
        self,
        scale_factor=4,
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
        flow_model="rapidflow",
        flow_ckpt="sintel",
        cpu_cache=False,

        num_cleaning_blocks=5,
        dynamic_refine_thres=255,
        is_sequential_cleaning=False,
        fix_cleaning=False,
        freeze_flow=False,

        cleaning_arch="swinir",
        cleaning_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.is_sequential_cleaning = bool(is_sequential_cleaning)
        self.dynamic_refine_thres = float(dynamic_refine_thres) / 255.0

        self.image_cleaning = self._build_cleaning_module(
            cleaning_arch=cleaning_arch,
            cleaning_kwargs=cleaning_kwargs,
            mid_channels=mid_channels,
            num_cleaning_blocks=num_cleaning_blocks,
        )

        if fix_cleaning:
            self.image_cleaning.requires_grad_(False)

        self.basicvsrpp = BasicVSRPlusPlus(
            scale_factor=scale_factor,
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            max_residue_magnitude=max_residue_magnitude,
            is_low_res_input=is_low_res_input,
            flow_model=flow_model,
            flow_ckpt=flow_ckpt,
            cpu_cache=cpu_cache,
        )

        if freeze_flow and hasattr(self.basicvsrpp, "flow_net"):
            self.basicvsrpp.flow_net.requires_grad_(False)
    

    def _build_cleaning_module(
        self,
        cleaning_arch,
        cleaning_kwargs: Optional[Dict[str, Any]],
        mid_channels,
        num_cleaning_blocks,
    ):
        arch = (cleaning_arch or "").strip().lower()
        overrides = dict(cleaning_kwargs or {})

        if arch == "realbasicvsr":
            defaults = {
                "mid_channels": mid_channels,
                "num_blocks": num_cleaning_blocks,
                "negative_slope": 0.1,
            }
            ctor = self._build_realbasicvsr_cleaning

        elif arch == "rrdb":
            defaults = {
                "in_ch": 3,
                "mid_ch": mid_channels,
                "out_ch": 3,
                "num_rrdb": num_cleaning_blocks,
                "rrdb_gc": 32,
            }
            ctor = CleaningModuleRRDB

        elif arch == "swinir":
            defaults = {
                "in_ch": 3,
                "embed_dim": 120,
                "img_size": 64,
                "patch_size": 1,
                "depths": [6] * num_cleaning_blocks,
                "num_heads": [6] * num_cleaning_blocks,
                "window_size": 8,
                "mlp_ratio": 2.0,
                "qkv_bias": True,
                "drop": 0.0,
                "attn_drop": 0.0,
                "drop_path": 0.1,
                "norm_layer": nn.LayerNorm,
                "ape": False,
                "patch_norm": True,
                "img_range": 1.0,
                "use_checkpoint": False,
            }
            ctor = CleaningModuleRSTB

        elif arch == "hat":
            defaults = {
                "img_size": 64,
                "patch_size": 1,
                "in_chans": 3,
                "embed_dim": 144,
                "depths": [6] * num_cleaning_blocks,
                "num_heads": [6] * num_cleaning_blocks,
                "window_size": 16,
                "compress_ratio": 24,
                "squeeze_factor": 24,
                "conv_scale": 0.01,
                "overlap_ratio": 0.5,
                "mlp_ratio": 2.0,
                "qkv_bias": True,
                "qk_scale": None,
                "drop_rate": 0.0,
                "attn_drop_rate": 0.0,
                "drop_path_rate": 0.1,
                "norm_layer": nn.LayerNorm,
                "ape": False,
                "patch_norm": True,
                "use_checkpoint": False,
                "img_range": 1.0,
            }
            ctor = CleaningModuleHAT

        else:
            raise ValueError(
                f"Invalid cleaning_arch='{cleaning_arch}'. "
                f"Choose one of: ['realbasicvsr','rrdb','swinir','hat']"
            )

        kwargs = dict(defaults)
        kwargs.update(overrides)

        if arch in ("hat", "swinir"):
            if "depths" in kwargs and "num_heads" in kwargs:
                if len(kwargs["num_heads"]) != len(kwargs["depths"]):
                    raise ValueError(f"{arch}: len(num_heads) must match len(depths)")

        self._validate_cleaning_kwargs(arch, kwargs)
        return ctor(**kwargs)


    def _build_realbasicvsr_cleaning(self, mid_channels, num_blocks, negative_slope=0.1):
        layers = [
            nn.Conv2d(3, mid_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlockNoBN(mid_channels))
        layers.append(nn.Conv2d(mid_channels, 3, 3, 1, 1, bias=True))
        return nn.Sequential(*layers)
    
    def _validate_cleaning_kwargs(self, arch, kwargs: Dict[str, Any]):
        if arch == "hat":
            sig = inspect.signature(CleaningModuleHAT.__init__)
        elif arch == "swinir":
            sig = inspect.signature(CleaningModuleRSTB.__init__)
        elif arch == "rrdb":
            sig = inspect.signature(CleaningModuleRRDB.__init__)
        elif arch == "realbasicvsr":
            sig = inspect.signature(self._build_realbasicvsr_cleaning)
        else:
            return

        valid = set(sig.parameters.keys())
        valid.discard("self")
        unknown = set(kwargs.keys()) - valid
        if unknown:
            raise ValueError(
                f"Unknown cleaning_kwargs for arch='{arch}': {sorted(unknown)}. "
                f"Valid keys include: {sorted(list(valid))[:30]}..."
            )


    def _clean_once(self, x_btchw):
        """x_btchw: (B*T, C, H, W); returns cleaned and residual."""
        r = self.image_cleaning(x_btchw)
        return x_btchw + r, r

    def forward(self, lqs, return_lqs=False):
        """
        lqs: (B, T, C, H, W)
        return_lqs: if True, also return the cleaned LR sequence for a cleaning loss
        """
        assert lqs.ndim == 5, "Expected (B,T,C,H,W)"
        B, T, C, H, W = lqs.shape

        # dynamic iterative cleaning (3 passes at most)
        lqs_clean = lqs.clone()
        residues_mean_abs = None
        for i in range(3):
            if self.is_sequential_cleaning:
                # iterate per time step to reduce memory
                residues = []
                for t in range(T):
                    yt, rt = self._clean_once(lqs_clean[:, t, :, :, :])
                    lqs_clean[:, t, :, :, :] = yt
                    residues.append(rt)
                residues = torch.stack(residues, dim=1)  # (B,T,C,H,W)
            else:
                x = lqs_clean.reshape(-1, C, H, W)
                y, r = self._clean_once(x)               # (B*T, C, H, W)
                lqs_clean = y.view(B, T, C, H, W)
                residues = r.view(B, T, C, H, W)

            residues_mean_abs = torch.mean(torch.abs(residues))
            if residues_mean_abs < self.dynamic_refine_thres:
                break

        # Super-resolution with the original backbone
        sr = self.basicvsrpp(lqs_clean)

        if return_lqs:
            return sr, lqs_clean
        return sr
