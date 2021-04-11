from .basicblocks import *
from argparse import Namespace

import torch
import torch.nn.functional as F


def sample_normal_jit(mu, log_var):
    sigma = torch.exp(log_var / 2)
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


class VSPM(nn.Module):
    '''
    kernel branch & weight branch
    '''

    def __init__(self, args, conv=default_conv):
        super(VSPM, self).__init__()
        self.scale = args.scale
        assert self.scale in [4, 8], f'sorry, {self.scale}X SR is not supported now'
        self.in_chan = args.in_chan
        self.n_chan = args.n_chan
        self.variational_z = args.variational_z  # variational z
        self.variational_w = args.variational_w
        self.alpha = args.alpha
        self.beta = args.beta  # parameters for gamma distribution
        self.upsample = args.upsample
        self.mode = args.mode

        s_z = 2 if self.variational_z else 1
        s_w = 2 if self.variational_w else 1
        if self.upsample:
            self.upsamplebranch = nn.Sequential(
                conv(3, 16, 3),
                nn.ReLU(inplace=True),
                conv(16, 3, 3)
            )

        # kernel branch
        kernel_branch_list = [conv(self.in_chan, self.n_chan // 4, 3),
                              nn.InstanceNorm2d(self.n_chan // 4),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(kernel_size=2, stride=2),

                              conv(self.n_chan // 4, self.n_chan // 2, 3),
                              nn.InstanceNorm2d(self.n_chan // 2),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(kernel_size=2, stride=2),

                              conv(self.n_chan // 2, self.n_chan, 3),
                              nn.InstanceNorm2d(self.n_chan),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(kernel_size=2, stride=2),

                              conv(self.n_chan, 2 * self.n_chan, 3),
                              nn.InstanceNorm2d(2 * self.n_chan),
                              nn.ReLU(inplace=True),
                              nn.MaxPool2d(kernel_size=2, stride=2),

                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten(),
                              nn.Linear(2 * self.n_chan, s_z * self.n_chan),
                              nn.Unflatten(1, (s_z * self.n_chan, 1, 1)),
                              nn.Conv2d(s_z * self.n_chan, s_z * self.n_chan, 2, 1, 1)]
        if self.scale == 4:
            kernel_branch_list += [nn.ConvTranspose2d(s_z * self.n_chan, s_z * self.n_chan, 3, 1),
                                   conv(s_z * self.n_chan, s_z * self.n_chan, 1)]
        elif self.scale == 8:
            kernel_branch_list += [nn.ConvTranspose2d(s_z * self.n_chan, s_z * self.n_chan, 3, 1),
                                   nn.ConvTranspose2d(s_z * self.n_chan, s_z * self.n_chan, 2, 2),
                                   conv(s_z * self.n_chan, s_z * self.n_chan, 1)]

        self.kernelbranch = nn.Sequential(*kernel_branch_list)

        self.weightbranch = nn.Sequential(
            conv(self.in_chan, self.n_chan // 4, 3),
            nn.GroupNorm(1, self.n_chan // 4),
            nn.ReLU(inplace=True),
            conv(self.n_chan // 4, self.n_chan // 2, 3),
            nn.GroupNorm(1, self.n_chan // 2),
            nn.ReLU(inplace=True),
            conv(self.n_chan // 2, self.n_chan, 3),
            nn.GroupNorm(1, self.n_chan),
            nn.ReLU(inplace=True),
            conv(self.n_chan, s_w * self.n_chan, 3),
            nn.GroupNorm(1, s_w * self.n_chan),
            nn.ReLU(inplace=True),
            conv(s_w * self.n_chan, s_w * self.n_chan, 1),
        )
        self.finalbranch = nn.Sequential(
            conv(3, 16, 3),
            nn.ReLU(inplace=True),
            conv(16, 16, 3),
            nn.ReLU(inplace=True),
            conv(16, 3, 3),
        )

    def generate_z(self, x):
        z = self.kernelbranch(x)
        if self.variational_z:
            mu_z, log_var_z = torch.chunk(z, 2, dim=1)
            z, _ = sample_normal_jit(mu_z, log_var_z)
            return z, mu_z, log_var_z
        else:
            return z

    def generate_w(self, x):
        weight = self.weightbranch(x)
        if self.variational_w:
            mu_w, log_var_w = torch.chunk(weight, 2, dim=1)
            weight, _ = sample_normal_jit(mu_w, log_var_w)
            return weight, mu_w, log_var_w
        else:
            return weight

    def forward(self, x):
        if self.upsample:
            upx = F.interpolate(x, scale_factor=self.scale, mode=self.mode)
            upx = self.upsamplebranch(upx)

        # 3 channels to 1 channel
        batchsize, channels, height, width = x.shape
        x = x.reshape(batchsize * channels, 1, height, width)

        # infer basis z & compute kl
        if self.variational_z:
            z, mu_z, log_var_z = self.generate_z(x)
            kl_z = mu_z * mu_z + torch.exp(log_var_z) - log_var_z
        else:
            z = self.generate_z(x)
            kl_z = None

        # sample w
        if self.variational_w:
            weight, mu_w, log_var_w = self.generate_w(x)
            mu_rou = (self.alpha + 0.5) / (self.beta + 0.5 * (mu_w * mu_w + torch.exp(log_var_w))).detach()
            kl_w = mu_rou * (mu_w * mu_w + torch.exp(log_var_w)) - log_var_w  # B X C X W X H
        else:
            weight = self.generate_w(x)
            kl_w = None

        # reshape for (SH) X (SW)
        weight = weight.reshape(batchsize * channels, self.n_chan, height * width)
        z = z.reshape(batchsize * channels, self.n_chan, self.scale * self.scale).permute(0, 2, 1)
        output = torch.matmul(z, weight).reshape(batchsize * channels, self.scale * self.scale, height, width)
        output = nn.PixelShuffle(self.scale)(output)

        # expand for RGB
        output = output.reshape(batchsize, channels, self.scale * height, self.scale * width)
        output = self.finalbranch(output)
        explore = output

        # upsampling
        if self.upsample:
            output += upx

        return output, kl_z, kl_w, explore


def make_vspm(scale=4, n_basis=64, n_color=1, alpha=3.0, beta=0.5,
              variational_z=False, variational_w=True, upsample=False, mode='bilinear'):
    args = Namespace()
    args.scale = scale
    args.n_chan = n_basis
    args.in_chan = n_color
    args.alpha = alpha
    args.beta = beta
    args.variational_z = variational_z
    args.variational_w = variational_w
    args.upsample = upsample
    args.mode = mode
    return VSPM(args)
