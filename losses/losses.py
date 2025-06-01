import torch
import torch.nn as nn
from torch.nn import functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)


class TVLoss(nn.Module):
    def __init__(self, weight=1e-4):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, img):
        tv_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
        tv_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        return self.weight * (tv_h + tv_w)


# From BasicSR
# https://github.com/XPixelGroup/BasicSR
# License: Apache-2.0
def get_local_weights(residual, ksize):
    """Get local weights for generating the artifact map of LDL.

    It is only called by the `get_refined_artifact_map` function.

    Args:
        residual (Tensor): Residual between predicted and ground truth images.
        ksize (Int): size of the local window.

    Returns:
        Tensor: weight for each pixel to be discriminated as an artifact pixel
    """

    pad = (ksize - 1) // 2
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')

    unfolded_residual = residual_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

# From BasicSR
# https://github.com/XPixelGroup/BasicSR
# License: Apache-2.0
def get_refined_artifact_map(img_gt, img_output, img_ema, ksize):
    """Calculate the artifact map of LDL
    (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022)

    Args:
        img_gt (Tensor): ground truth images.
        img_output (Tensor): output images given by the optimizing model.
        img_ema (Tensor): output images given by the ema model.
        ksize (Int): size of the local window.

    Returns:
        overall_weight: weight for each pixel to be discriminated as an artifact pixel
        (calculated based on both local and global observations).
    """

    residual_ema = torch.sum(torch.abs(img_gt - img_ema), 1, keepdim=True)
    residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

    patch_level_weight = torch.var(residual_sr.clone(), dim=(-1, -2, -3), keepdim=True)**(1 / 5)
    pixel_level_weight = get_local_weights(residual_sr.clone(), ksize)
    overall_weight = patch_level_weight * pixel_level_weight

    overall_weight[residual_sr < residual_ema] = 0

    return overall_weight
