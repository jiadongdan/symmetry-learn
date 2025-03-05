import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_kernels(n_max, size):
    zps = ZPs(n_max=n_max, size=size)
    kernels = zps.polynomials
    mask1 = zps.m > 1
    mask2 = zps.m < -1
    A_kernels = kernels[mask1]
    B_kernels = kernels[mask2]
    return A_kernels, B_kernels

def compute_tensor_weights(n_max, size):
    theta = np.linspace(0, 2 * np.pi, 361)[0:360]
    ms = zm.to_complex().m
    cosmt = np.array([np.cos(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
    sinmt = np.array([np.sin(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
    matrix = np.hstack([cosmt, sinmt])  # 360 x 60



class FixedConvLayer(nn.Module):
    def __init__(self, kernels: torch.Tensor, linear_weight: torch.Tensor, stride=1):
        super(FixedConvLayer, self).__init__()
        assert kernels.shape[1] % 2 == 1 and kernels.shape[2] % 2 == 1, "Kernel size must be odd to preserve input shape."
        self.register_buffer('kernels', kernels.unsqueeze(1))  # Shape becomes (N, 1, H, W)
        self.stride = stride
        self.kernel_size = (kernels.shape[1], kernels.shape[2])  # (H, W)
        self.padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)  # (pad_H, pad_W)
        self.norm_factor = 4./(self.kernel_size[0] * self.kernel_size[1]) / np.pi  # normalizingh factor

        # Linear layer with fixed weight
        self.register_buffer('linear_weight', linear_weight)  # Store as non-trainable

    def forward(self, x):
        x = x.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
        x1 = self.norm_factor * F.conv2d(x, self.kernels, stride=self.stride, padding=self.padding).squeeze(0)
        x2 = x1 ** 2

        # Apply fixed linear transformation along first axis
        x3 = torch.matmul(self.linear_weight, x2.view(x2.shape[0], -1))  # (M, N) @ (N, H*W) -> (M, H*W)
        x3 = x3.view(-1, x2.shape[1], x2.shape[2])  # Reshape back to (M, H, W)

        return x3