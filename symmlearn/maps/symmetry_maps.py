import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_rot_maps_matrix(n_folds, m):
    """
    Constructs a matrix where each row is a transformation based on the array `m`.
    Each element in `n_folds` corresponds to a specific condition in `m`.

    Parameters:
        n_folds (array-like): Indices used for certain conditions.
        m (array-like): Array of markers to determine placement in matrix.

    Returns:
        numpy.ndarray: Matrix of transformations.
    """
    # Ensure inputs are 1D numpy arrays
    n_folds = check_array1d(n_folds)
    m = check_array1d(m)
    m = np.abs(m)

    # Initialize the matrix with zeros
    matrix = np.zeros((len(n_folds), len(m)))

    for i, n_fold in enumerate(n_folds):
        # Create a row initialized to zero
        # Do NOT use np.zeros_like
        row = np.zeros(len(m))
        is_current_fold = (m % n_fold == 0) & (m > 1)
        is_special_case = (m == 0) | (m == 1)
        not_covered = ~(is_current_fold | is_special_case)

        # Set conditions based on `m`
        row[is_current_fold] = 1
        row[not_covered] = -1. / (n_fold - 1) if n_fold > 1 else 0  # Avoid division by zero

        # Place the row in the matrix
        matrix[i] = row

    return matrix


def compute_kernels_and_linear_weights(n_max, size, n_folds=[2, 3, 4, 6]):
    zps = ZPs(n_max=n_max, size=size)
    kernels = zps.polynomials
    linear_weights = construct_rot_maps_matrix(n_folds, zps.m)
    return kernels, linear_weights

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
        if x.dim() == 2:  # Single image case (H, W)
            x = x.unsqueeze(0).unsqueeze(0)  # Shape becomes (1, 1, H, W)
        elif x.dim() == 3:  # Stack of images (num_imgs, H, W)
            x = x.unsqueeze(1)  # Shape becomes (num_imgs, 1, H, W)
        else:
            raise ValueError("Input must be of shape (H, W) or (num_imgs, H, W)")

        x1 = self.norm_factor * F.conv2d(x, self.kernels, stride=self.stride, padding=self.padding)
        x2 = x1 ** 2    # Shape becomes (num_imgs, num_kernels, H, W)

        # Reshape x2 to (num_imgs, num_kernels, H*W)
        x2_flat = x2.view(x2.shape[0], x2.shape[1], -1)  # (num_imgs, num_kernels, H*W)

        # Apply linear transformation: (another_num, num_kernels) @ (num_imgs, num_kernels, H*W) -> (num_imgs, another_num, H*W)
        x3 = torch.matmul(self.linear_weight, x2_flat)  # (num_imgs, another_num, H*W)

        # Reshape back to (num_imgs, another_num, H, W)
        x3 = x3.view(x2.shape[0], -1, x2.shape[2], x2.shape[3])

        return x3

