import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, strides=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, strides,
        padding=(kernel_size // 2), bias=bias)

