# Adapted from:  BasicVSR_PlusPlus (https://github.com/ckkelvinchan/BasicVSR_PlusPlus)
# Apache License 2.0
#
# This file has been modified for use in the PAWS project.
#
# Original copyright:
# Copyright (c) 2020 MMEditing Authors. Licensed under the Apache License, Version 2.0.


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d, deform_conv2d

import ptlflow


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                         f'flow ({flow.size()[1:3]}) are not the same.')
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    # torch.meshgrid has been modified in 1.10.0 (compatibility with previous
    # versions), and will be further modified in 1.12 (Breaking Change)
    if 'indexing' in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing='ij')
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad = False

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    grid_flow = grid_flow.type(x.type())
    output = F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
    return output

class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.

    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)

    def forward(self, x):
        """Forward function for PixelShufflePack.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class BasicVSRPlusPlus(nn.Module):
    """
    BasicVSR++

    Implements a recurrent architecture for video super-resolution with enhanced
    propagation and second-order deformable alignment, supporting arbitrary positive
    power-of-two upsampling factors.

    Reference:
        Chan et al., "BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment."
        https://arxiv.org/abs/2104.13371

    Args:
        scale_factor (int):
            Upsampling ratio. Must be a positive power of two (e.g., 1, 2, 4, 8, 16).
        mid_channels (int, optional):
            Number of channels in intermediate feature representations. Default: 64.
        num_blocks (int, optional):
            Number of residual blocks in each propagation branch. Default: 7.
        max_residue_magnitude (int, optional):
            Maximum allowed magnitude for alignment offset residues (see Eq. 6 in the reference).
            Default: 10.
        is_low_res_input (bool, optional):
            If True, the model expects inputs as low-resolution frames and applies upsampling,
            adding a bilinear skip connection. If False, inputs are assumed to be high-resolution.
            They are internally downsampled, and the original frames are used as a skip connection.
            Default: True.
        flow_model (str, optional):
            Name of the optical flow model to use (must be present in PTLFlow).
        flow_ckpt (str, optional):
            Checkpoint identifier or path for the optical flow model.
        cpu_cache (bool, optional):
            Whether to cache features on CPU to reduce GPU memory usage, at the cost of slower
            inference. Default: False.

    Notes:
        - The model supports only positive power-of-two upsampling factors.
        - Optical flow is computed with a user-specified PTLFlow model.
        - Setting `cpu_cache=True` enables CPU feature caching; use with care, as it may degrade throughput.
    """

    def __init__(
        self,
        scale_factor=4,
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
        flow_model="rapidflow",
        flow_ckpt="sintel",
        cpu_cache=False
    ):
        super().__init__()
        assert scale_factor > 0 and (scale_factor & (scale_factor - 1)) == 0, "scale_factor must be a positive power of 2"
        self.scale_factor = scale_factor
        self.mid_channels = mid_channels
        self.is_low_res_input = is_low_res_input
        self.cpu_cache = cpu_cache

        # validate flow model name
        available_flows = ptlflow.get_model_names()
        if flow_model not in available_flows:
            raise ValueError(
                f"Optical flow model '{flow_model}' is not supported.\n"
                f"Available options: {available_flows}"
            )

        # optical flow
        self.flow_net = ptlflow.get_model(flow_model, ckpt_path=flow_ckpt)

        # feature extraction module
        if is_low_res_input:
            layers = []
            layers.append(nn.Conv2d(3, mid_channels, 3, 1, 1, bias=True))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            for _ in range(5):
                layers.append(ResidualBlockNoBN(mid_channels))
            self.feat_extract  = nn.Sequential(*layers)
        else:
            layers = []
            layers.append(nn.Conv2d(3, mid_channels, 3, 2, 1))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            layers.append(nn.Conv2d(mid_channels, mid_channels, 3, 2, 1))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            for _ in range(5):
                layers.append(ResidualBlockNoBN(mid_channels))
            self.feat_extract  = nn.Sequential(*layers)

        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ["backward_1", "forward_1", "backward_2", "forward_2"]
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude,
            )
            
            layers = []
            layers.append(nn.Conv2d((2 + i) * mid_channels, mid_channels, 3, 1, 1, bias=True))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            for _ in range(num_blocks):
                layers.append(ResidualBlockNoBN(mid_channels))
            self.backbone[module]  = nn.Sequential(*layers)

        # upsampling module
        layers = []
        layers.append(nn.Conv2d(5 * mid_channels, mid_channels, 3, 1, 1, bias=True))
        layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        for _ in range(5):
            layers.append(ResidualBlockNoBN(mid_channels))
        self.reconstruction  = nn.Sequential(*layers)

        n_stages = int(math.log2(scale_factor))
        self.upsample_stages = nn.ModuleList([
            PixelShufflePack(mid_channels, mid_channels, 2, upsample_kernel=3)
            for _ in range(n_stages)
        ])

        self.conv_hr = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(mid_channels, 3, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False
        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lrs):
        """
        Compute optical flow between consecutive frames using PTLFlow.
        """
        n, t, c, h, w = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)

        inputs = torch.stack([lrs_1, lrs_2], dim=1)
        predictions = self.flow_net({'images': inputs})
        flow_pred = predictions['flows'].squeeze(1)
        flows_backward = flow_pred.view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:
            flows_forward = None
        else:
            # For forward flow, swap the image order.
            inputs = torch.stack([lrs_2, lrs_1], dim=1)
            predictions = self.flow_net({'images': inputs})
            flow_pred = predictions['flows'].squeeze(1)
            flows_forward = flow_pred.view(n, t - 1, 2, h, w)
        
        if self.cpu_cache:
            flows_backward = flows_backward.cpu()
            flows_forward = flows_forward.cpu()

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats["spatial"])))
        mapping_idx += mapping_idx[::-1]

        if "backward" in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats["spatial"][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()
            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # concatenate and residual blocks
            feat = [feat_current] + [feats[k][idx] for k in feats if k not in ["spatial", module_name]] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if "backward" in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs, feats):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propagation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, scale_factor*h, scale_factor*w).
        """

        outputs = []
        num_outputs = len(feats["spatial"])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != "spatial"]
            hr.insert(0, feats["spatial"][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)

            for up in self.upsample_stages:
                hr = self.lrelu(up(hr))

            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)
            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def forward(self, lqs):
        """Forward function for BasicVSR++.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, scale_factor*h, scale_factor*w).
        """

        n, t, c, h, w = lqs.size()

        if self.is_low_res_input:
            lqs_downsample = lqs.clone()
        else:
            down = 1.0 / self.scale_factor
            new_h, new_w = int(h * down), int(w * down)
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w),
                size=(new_h, new_w),
                mode='bicubic',
                align_corners=False
            ).view(n, t, c, new_h, new_w)

        
        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}
        # compute spatial features
        if self.cpu_cache:
            feats["spatial"] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs[:, i, :, :, :]).cpu()
                feats["spatial"].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs.view(-1, c, h, w))
            h, w = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h, w)
            feats["spatial"] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            "The height and width of low-res inputs must be at least 64, " f"but got {h} and {w}."
        )
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # feature propagation
        for iter_ in [1, 2]:
            for direction in ["backward", "forward"]:
                module = f"{direction}_{iter_}"

                feats[module] = []

                if direction == "backward":
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        return self.upsample(lqs, feats)


class SecondOrderDeformableAlignment(DeformConv2d):
    """Second-order deformable alignment module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        max_residue_magnitude (int): The maximum magnitude of the offset
            residue (Eq. 6 in paper). Default: 10.
        deform_groups (int, optional): Number of deformable groups. Default: 16.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop("max_residue_magnitude", 10)
        self.deform_groups = kwargs.pop("deform_groups", 16)

        super(SecondOrderDeformableAlignment, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()
    
    def init_offset(self):
        """
        Initialize the final convolutional layer in `conv_offset` to produce zero offsets and mask.
        """
        layer = self.conv_offset[-1]
        if hasattr(layer, "weight") and layer.weight is not None:
            nn.init.constant_(layer.weight, 0)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x, extra_feat, flow_1, flow_2):
        """
        Compute aligned features using second-order deformable convolution.

        Args:
            x (Tensor): Input features of shape (N, C, H, W).
            extra_feat (Tensor): Auxiliary features concatenated with flows, shape (N, C', H, W).
            flow_1 (Tensor): First-order optical flow, shape (N, 2, H, W).
            flow_2 (Tensor): Second-order optical flow, shape (N, 2, H, W).

        Returns:
            Tensor: Output features after deformable alignment, shape (N, out_channels, H, W).
        """
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # offset
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        # mask
        mask = torch.sigmoid(mask)

        return deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding, self.dilation, mask)
