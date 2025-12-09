import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mtflearn.features import ZPs
from mtflearn.features import nm2j


def compute_kernels_weights(n_max, size):
    zps = ZPs(n_max=n_max, size=size)
    kernels = zps.polynomials
    mask = zps.m > 1
    inds= nm2j(zps.n[mask], -zps.m[mask])
    A_kernels = kernels[mask]
    B_kernels = kernels[inds]
    kernels = np.vstack([A_kernels, B_kernels])

    # compute weights
    theta = np.linspace(0, 2 * np.pi, 361)[0:360]
    ms = zps.m[mask]
    cosmt = np.array([np.cos(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
    sinmt = np.array([np.sin(m * t) for t in theta for m in ms]).reshape(len(theta), -1)
    weights = np.hstack([cosmt, sinmt])  # 360 x 50 if n_max = 10
    return kernels, weights


class RefMap(nn.Module):
    def __init__(self, kernels: torch.Tensor, linear_weight: torch.Tensor, stride=1):
        super(RefMap, self).__init__()
        num_kernels = kernels.shape[0]
        kernels_A = kernels[0:num_kernels // 2]
        kernels_B = kernels[num_kernels //2 :]
        assert kernels.shape[1] % 2 == 1 and kernels.shape[2] % 2 == 1, "Kernel size must be odd to preserve input shape."
        self.register_buffer('kernels_A', kernels_A.unsqueeze(1))  # Shape becomes (N, 1, H, W)
        self.register_buffer('kernels_B', kernels_B.unsqueeze(1))  # Shape becomes (N, 1, H, W)
        self.stride = stride
        self.kernel_size = (kernels.shape[1], kernels.shape[2])  # (H, W)
        self.padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)  # (pad_H, pad_W)
        self.norm_factor = 4./(self.kernel_size[0] * self.kernel_size[1]) / np.pi  # normalizingh factor

        # Linear layer with fixed weight
        self.register_buffer('linear_weight', linear_weight)  # Store as non-trainable

    def forward(self, x, return_angle=False):
        if x.dim() == 2:  # Single image case (H, W)
            x = x.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
        elif x.dim() == 3:  # Stack of images (num_imgs, H, W)
            x = x.unsqueeze(1)  # Shape becomes (num_imgs, 1, H, W)
        else:
            raise ValueError("Input must be of shape (H, W) or (num_imgs, H, W)")

        A = self.norm_factor * F.conv2d(x, self.kernels_A, stride=self.stride, padding=self.padding) # Shape becomes (num_imgs, 25, H, W)
        B = self.norm_factor * F.conv2d(x, self.kernels_B, stride=self.stride, padding=self.padding) # Shape becomes (num_imgs, 25, H, W)
        x_part1 = A**2 - B**2
        x_part2 = 2 * A * B
        # Concatenate x_part1 and x_part2 along the second axis
        x2 = torch.cat([x_part1, x_part2], dim=1)

        # Normalize x2 along the second axis, why I set p=1?
        x2 = F.normalize(x2, p=1, dim=1)

        # Reshape x2 to (num_imgs, num_kernels, H*W)
        x2_flat = x2.view(x2.shape[0], x2.shape[1], -1)

        # Apply linear transformation: (another_num, num_kernels) @ (num_imgs, num_kernels, H*W) -> (num_imgs, another_num, H*W)
        x3 = torch.matmul(self.linear_weight, x2_flat)  # (num_imgs, another_num, H*W)

        # Reshape back to (num_imgs, another_num, H, W)
        x3 = x3.view(x2.shape[0], -1, x2.shape[2], x2.shape[3])

        x3_max, max_inds = torch.max(x3, dim=1)

        if return_angle:
            theta = np.linspace(0, 2 * np.pi, 361)[0:360]
            theta_map = theta[max_inds.cpu()]

            return x3_max, theta_map / 2.0
        else:
            return x3_max