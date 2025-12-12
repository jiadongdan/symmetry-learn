import numpy as np
from typing import Tuple, Optional, Union

def stratified_sampling(
        size: Union[float, Tuple[float, float]],
        seed: Optional[int] = None,
        n_samples: Optional[int] = None,
        n_samples_per_axis: Optional[Tuple[int, int]] = None,
        patch_size: Optional[float] = None,
        overlap: float = 0.0,
        jitter: bool = True
) -> np.ndarray:
    """Stratified sampling in 2D spatial region."""
    if seed is not None:
        np.random.seed(seed)

    # Parse size into (height, width)
    if isinstance(size, (int, float)):
        height, width = float(size), float(size)
    else:
        height, width = float(size[0]), float(size[1])

    # Determine grid dimensions
    provided_params = sum([
        n_samples is not None,
        n_samples_per_axis is not None,
        patch_size is not None
    ])

    if provided_params == 0:
        raise ValueError("Must provide one of: n_samples, n_samples_per_axis, or patch_size")
    elif provided_params > 1:
        raise ValueError("Cannot use n_samples, n_samples_per_axis, and patch_size together")

    if patch_size is not None:
        # Patch-based sampling with square overlap
        if overlap < 1.0:
            stride = patch_size * (1.0 - overlap)
        else:
            stride = patch_size - overlap
            if stride <= 0:
                raise ValueError(f"Overlap {overlap} is too large for patch_size {patch_size}")

        n_y = int(np.ceil(height / stride))
        n_x = int(np.ceil(width / stride))
        n_y = max(1, n_y)
        n_x = max(1, n_x)

        cell_height = stride
        cell_width = stride

    elif n_samples_per_axis is not None:
        n_y, n_x = n_samples_per_axis
        cell_height = height / n_y
        cell_width = width / n_x

    else:  # n_samples is not None
        aspect_ratio = width / height
        n_y = int(np.sqrt(n_samples / aspect_ratio))
        n_x = int(np.sqrt(n_samples * aspect_ratio))

        while n_y * n_x < n_samples:
            if n_x * height < n_y * width:
                n_x += 1
            else:
                n_y += 1

        cell_height = height / n_y
        cell_width = width / n_x

    # Generate stratified samples
    points = []

    for i in range(n_y):
        for j in range(n_x):
            y_min = i * cell_height
            x_min = j * cell_width

            if jitter:
                y = y_min + np.random.rand() * cell_height
                x = x_min + np.random.rand() * cell_width
            else:
                y = y_min + cell_height / 2
                x = x_min + cell_width / 2

            points.append([x, y])

    return np.array(points) # Returns (x, y) coordinates


