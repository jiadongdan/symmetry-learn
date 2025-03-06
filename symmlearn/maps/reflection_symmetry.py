import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kernels_weights(n_max, size):
    zps = ZPs(n_max=n_max, size=size)
    kernels = zps.polynomials
    mask = zps.m > 1
    inds= nm2j(zps.n[mask], -zps.m[mask])
    A_kernels = kernels[mask]
    B_kernels = kernels[inds]

    # compute weights
    theta = np.linspace(0, 2 * np.pi, 361)[0:360]
    ms = zps.m[mask]
    cosmt = np.array([np.cos(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
    sinmt = np.array([np.sin(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
    weights = np.hstack([cosmt, sinmt])  # 360 x 60
    return A_kernels, B_kernels, weights


class FixedConvLayer(nn.Module):
    def __init__(self, kernels: torch.Tensor, linear_weight: torch.Tensor, stride=1):
        super(FixedConvLayer, self).__init__()
        num_kernels = kernels.shape[0]
        kernels_A = kernels[0:num_kernels // 2]
        kernels_B = kernels[num_kernels //2 :]
        assert kernels.shape[1] % 2 == 1 and kernels.shape[2] % 2 == 1, "Kernel size must be odd to preserve input shape."
        self.register_buffer('kernels', kernels_A.unsqueeze(1))  # Shape becomes (N, 1, H, W)
        self.register_buffer('kernels', kernels_B.unsqueeze(1))  # Shape becomes (N, 1, H, W)
        self.stride = stride
        self.kernel_size = (kernels.shape[1], kernels.shape[2])  # (H, W)
        self.padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)  # (pad_H, pad_W)
        self.norm_factor = 4./(self.kernel_size[0] * self.kernel_size[1]) / np.pi  # normalizingh factor

        # Linear layer with fixed weight
        self.register_buffer('linear_weight', self.linear_weight)  # Store as non-trainable

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
        A = self.norm_factor * F.conv2d(x, self.kernels_A, stride=self.stride, padding=self.padding).squeeze(0)
        B = self.norm_factor * F.conv2d(x, self.kernels_B, stride=self.stride, padding=self.padding).squeeze(0)
        x_part1 = A**2 - B**2
        x_part2 = 2 * A * B
        # concatenate part1 and part2

        return x_part1