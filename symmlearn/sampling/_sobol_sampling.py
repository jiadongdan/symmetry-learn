import numpy as np
from scipy.stats import qmc
from typing import Union, Tuple, Optional

def sobol_sampling(size, seed, n_samples=None, patch_size=None, overlap=0):
    if n_samples is None:
        if patch_size is not None:
            if not isinstance(size, tuple):
                raise ValueError("size must be (height, width) when using patch_size")

            height, width = size

            if isinstance(patch_size, int):
                patch_h, patch_w = patch_size, patch_size
            else:
                patch_h, patch_w = patch_size

            if overlap < 1.0:
                stride_h = patch_h * (1.0 - overlap)
                stride_w = patch_w * (1.0 - overlap)
            else:
                stride_h = patch_h - overlap
                stride_w = patch_w - overlap

            n_y = int(np.ceil(height / stride_h))
            n_x = int(np.ceil(width / stride_w))
            n_samples = n_y * n_x

        elif isinstance(size, int):
            n_samples = size
            height = width = size
        else:
            height, width = size
            n_samples = height * width

    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    samples = sampler.random(n=n_samples)

    # Scale to (height, width) and return as (x, y)
    samples[:, 0] *= width   # Scale first column to width (x)
    samples[:, 1] *= height  # Scale second column to height (y)

    return samples  # Returns (x, y) coordinates