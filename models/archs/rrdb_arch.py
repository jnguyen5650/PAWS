# Adapted from BasicSR RRDB implementation:
# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/rrdbnet_arch.py
#
# Licensed under the Apache License, Version 2.0
#
# This file has been modified for use in the PAWS project.
#
# Original copyright:
# Copyright (c) 2018-2022 BasicSR Authors


import torch
import torch.nn as nn


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class CleaningModuleRRDB(nn.Module):
    def __init__(self, in_ch=3, mid_ch=64, out_ch=3, num_rrdb=4, rrdb_gc=32):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=True)

        body = []
        for _ in range(num_rrdb):
            body.append(RRDB(mid_ch, gc=rrdb_gc))
        self.body = nn.Sequential(*body)
        self.trunk_conv = nn.Conv2d(mid_ch, mid_ch, 3, 1, 1, bias=True)
        self.proj_out = nn.Conv2d(mid_ch, out_ch, 3, 1, 1, bias=True)

    def forward(self, x):
        f0 = self.conv_in(x)
        fb = self.body(f0)
        f  = f0 + self.trunk_conv(fb)
        r  = self.proj_out(f)
        return r
