import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys

from utils import get_grid


def conv(batch_norm, in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )


class WarpingLayer(nn.Module):

    def __init__(self, args):
        super(WarpingLayer, self).__init__()
        self.args = args

    def forward(self, x, flow):
        args = self.args
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow

        # F.grid_sample接受的参数是（被采样图，normalized采样点）
        # 所以这里先normalize flow, return x_warp = x2_norm + flow_norm\

        # 奇怪的是：原文说的是，把I2朝着I1进行warp.却是x2+flow=x1?
        # 也就是说这个flow是x2->x1的flow？第一张图和第二张图的顺序？？？

        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:, 0, :, :] = flow[:, 0, :, :] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:, 1, :, :] = flow[:, 1, :, :] / ((flow.size(2) - 1.0) / 2.0)

        grid = (get_grid(x).to(args.device) + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


class CostVolumeLayer(nn.Module):
    def __init__(self, args):
        super(CostVolumeLayer, self).__init__()
        self.args = args
        self.search_range = args.search_range

    def forward(self, x1, x2):
        """
            from [Accurate Optical Flow via Direct Cost Volume Processing]
            Q:  import numpy as np
                f=np.array(x1[0,:,1,1])
                print(np.sum(f*f))
                output:0.050772406, 0.038208075, 0.020489007.
            x1,x2 here is not normlized?
        :param x1: must be normlized, that is ||x1||^2 == 1
        :param x2: same with up
        :return:
        """
        args = self.args

        shape = list(x1.size());
        shape[1] = (self.search_range * 2 + 1) ** 2
        cv = torch.zeros(shape).to(args.device)

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                if i < 0:
                    slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0:
                    slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:
                    slice_h, slice_h_r = slice(None), slice(None)

                if j < 0:
                    slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0:
                    slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:
                    slice_w, slice_w_r = slice(None), slice(None)

                cv[:, (self.search_range * 2 + 1) * i + j, slice_h, slice_w] = (
                        x1[:, :, slice_h, slice_w] * x2[:, :, slice_h_r, slice_w_r]).sum(1)
        return cv / shape[1]


class FeaturePyramidExtractor(nn.Module):

    def __init__(self, args):
        super(FeaturePyramidExtractor, self).__init__()
        self.args = args

        self.convs = []
        for l, (ch_in, ch_out) in enumerate(zip(args.lv_chs[:-1], args.lv_chs[1:])):
            layer = nn.Sequential(
                conv(args.batch_norm, ch_in, ch_out, stride=2),
                conv(args.batch_norm, ch_out, ch_out)
            )
            self.add_module(f'Feature(Lv{l})', layer)
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x);
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class OpticalFlowEstimator(nn.Module):

    def __init__(self, args, ch_in):
        super(OpticalFlowEstimator, self).__init__()
        self.args = args

        self.convs = nn.Sequential(
            conv(args.batch_norm, ch_in, 128),
            conv(args.batch_norm, 128, 128),
            conv(args.batch_norm, 128, 96),
            conv(args.batch_norm, 96, 64),
            conv(args.batch_norm, 64, 32),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                      bias=True)
        )

    def forward(self, x):
        return self.convs(x)


class ContextNetwork(nn.Module):

    def __init__(self, args, ch_in):
        super(ContextNetwork, self).__init__()
        self.args = args

        self.convs = nn.Sequential(
            conv(args.batch_norm, ch_in, 128, 3, 1, 1),
            conv(args.batch_norm, 128, 128, 3, 1, 2),
            conv(args.batch_norm, 128, 128, 3, 1, 4),
            conv(args.batch_norm, 128, 96, 3, 1, 8),
            conv(args.batch_norm, 96, 64, 3, 1, 16),
            conv(args.batch_norm, 64, 32, 3, 1, 1),
            conv(args.batch_norm, 32, 2, 3, 1, 1)
        )

    def forward(self, x):
        return self.convs(x)
