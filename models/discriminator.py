import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from .registry import register_discriminator


class TemporalBlock(nn.Module):
    """
    Temporal block that applies a 1D convolution over the temporal dimension
    and then a temporal self-attention. Input shape is [B, T, C, H, W].
    """
    def __init__(self, channels, kernel_size=3, num_heads=4):
        super().__init__()
        self.temporal_conv = spectral_norm(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False)
        )
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # [B, T, C, H, W] -> [B, H, W, C, T] -> [B*H*W, C, T]
        x = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, C, T)

        x = self.temporal_conv(x)  

        # Prepare for self attention: [N, T, C]
        x = x.permute(0, 2, 1).contiguous()  # [B*H*W, T, C]

        y = self.ln(x)
        attn_out, _ = self.attn(y, y, y)
        x = x + attn_out # residual connection

        # Back to [B, T, C, H, W]
        x = x.permute(0, 2, 1).contiguous().view(B, H, W, C, T)
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        return x


@register_discriminator()
class SpatioTemporalUNetDiscriminator(nn.Module):
    """
    U-Net discriminator with multi-scale spatiotemporal (temporal attention) blocks.

    Input: [B, T, C, H, W], output: [B, 1, H, W]
    """
    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm

        # Encoder
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat*2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat*2, num_feat*4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat*4, num_feat*8, 4, 2, 1, bias=False))

        # Temporal modules on encoder
        self.temp1 = TemporalBlock(num_feat*2)
        self.temp2 = TemporalBlock(num_feat*4)
        self.temp3 = TemporalBlock(num_feat*8)

        # Decoder
        self.conv4 = norm(nn.Conv2d(num_feat*8, num_feat*4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat*4, num_feat*2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat*2, num_feat,   3, 1, 1, bias=False))

        # Temporal modules on decoder
        self.temp4 = TemporalBlock(num_feat*4)
        self.temp5 = TemporalBlock(num_feat*2)

        # Final convs
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Flatten for 2D convs
        x = x.view(B*T, C, H, W)

        # Encoder
        x0 = F.leaky_relu(self.conv0(x), 0.2)
        x1 = F.leaky_relu(self.conv1(x0), 0.2)  # [B*T, f*2, H/2, W/2]
        x2 = F.leaky_relu(self.conv2(x1), 0.2)  # [B*T, f*4, H/4, W/4]
        x3 = F.leaky_relu(self.conv3(x2), 0.2)  # [B*T, f*8, H/8, W/8]

        # Multi-scale temporal attention on encoder outputs
        # Level 1
        Bx, C1, H1, W1 = x1.shape
        t1 = x1.view(B, T, C1, H1, W1)
        t1 = self.temp1(t1)
        x1 = t1.view(B*T, C1, H1, W1)

        # Level 2
        Bx, C2, H2, W2 = x2.shape
        t2 = x2.view(B, T, C2, H2, W2)
        t2 = self.temp2(t2)
        x2 = t2.view(B*T, C2, H2, W2)

        # Level 3 (bottleneck)
        Bx, C3, H3, W3 = x3.shape
        t3 = x3.view(B, T, C3, H3, W3)
        t3 = self.temp3(t3)
        x3 = t3.view(B*T, C3, H3, W3)

        # Decoder
        x4 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x4), 0.2)
        if self.skip_connection:
            x4 = x4 + x2
        # Temporal at decoder level 1
        Bx, C4, H4, W4 = x4.shape
        t4 = x4.view(B, T, C4, H4, W4)
        t4 = self.temp4(t4)
        x4 = t4.view(B*T, C4, H4, W4)

        x5 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x5), 0.2)
        if self.skip_connection:
            x5 = x5 + x1
        # Temporal at decoder level 2
        Bx, C5, H5, W5 = x5.shape
        t5 = x5.view(B, T, C5, H5, W5)
        t5 = self.temp5(t5)
        x5 = t5.view(B*T, C5, H5, W5)

        x6 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x6), 0.2)
        if self.skip_connection:
            x6 = x6 + x0

        out = F.leaky_relu(self.conv7(x6), 0.2)
        out = F.leaky_relu(self.conv8(out), 0.2)
        out = self.conv9(out)  # [B*T,1,H,W]

        # Reshape
        out = out.view(B, T, 1, out.shape[2], out.shape[3])
        # Ablation: average over time
        # out = out.mean(dim=1)
        return out


# From BasicSR
# https://github.com/XPixelGroup/BasicSR
# License: Apache-2.0
@register_discriminator()
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Flatten for 2D convs
        x = x.view(B*T, C, H, W)
        
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

        return out
