import torch
import numpy as np
from typing import Optional, Sequence

from .rotational_symmetry import compute_kernels_and_linear_weights
from .rotational_symmetry import RotMaps
from .reflection_symmetry import compute_kernels_weights
from .reflection_symmetry import RefMap

def normalize_array(data, vmin=-1., vmax=1.):
    return (data - data.min())(vmax - vmin)/(data.max() - data.min())

def get_rot_maps_(img, patch_size, n_max=10, n_folds=[2, 3, 4, 6], device=None, normalize=False):
    kernels, weights = compute_kernels_and_linear_weights(n_max=n_max, size=patch_size, n_folds=n_folds)

    # Check for GPU availability
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # numpy array to tensor
    torch_kernels = torch.from_numpy(kernels).float().to(device)
    torch_weights = torch.from_numpy(weights).float().to(device)

    torch_img = torch.from_numpy(img).float().to(device)

    rotmaps = RotMaps(torch_kernels, torch_weights).to(device)

    output = rotmaps(torch_img)
    if normalize:
        output = normalize(output)
    return output[0, :, :, :].cpu().numpy()

def get_rot_maps(
        img: np.ndarray,
        patch_size: int,
        n_max: int = 10,
        n_folds: Sequence[int] = (2, 3, 4, 6),
        device: Optional[torch.device] = None,
        normalize_output: bool = False
) -> np.ndarray:
    """
    Compute rotation maps for a single 2D image.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image, shape (H, W).
    patch_size : int
        Patch size to pass to `compute_kernels_and_linear_weights`.
    n_max : int, optional
        Maximum rotation order (default: 10).
    n_folds : Sequence[int], optional
        List of fold values to pass to `compute_kernels_and_linear_weights` (default: (2, 3, 4, 6)).
    device : torch.device or None, optional
        If None, automatically chooses CUDA if available, else CPU.
    normalize_output : bool, optional
        If True, apply min–max normalization per rotation map so that each map lies in [0, 1].

    Returns
    -------
    np.ndarray
        An array of shape (N_maps, H, W), where N_maps is determined by your RotMaps module.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Compute kernels & weights (assumed to return NumPy arrays)
    kernels, weights = compute_kernels_and_linear_weights(
        n_max=n_max,
        size=patch_size,
        n_folds=tuple(n_folds)
    )

    # 2) Move kernels & weights to device as float32 tensors
    kernels_t = torch.as_tensor(kernels, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

    # 3) Convert the 2D img to a tensor of shape (1, 1, H, W)
    if img.ndim != 2:
        raise ValueError(f"Expected img.ndim == 2, but got {img.ndim}")
    img_t = torch.as_tensor(img, dtype=torch.float32, device=device)

    # 4) Build the RotMaps module and put it into eval mode
    rotmaps_model = RotMaps(kernels_t, weights_t).to(device)
    rotmaps_model.eval()

    # 5) Forward pass under no_grad
    with torch.no_grad():
        # Output shape: (1, N_maps, H, W)
        output = rotmaps_model(img_t)

        if normalize_output:
            # Flatten over all maps and pixels to compute a single global min and max
            flat = output.view(-1)
            global_min = flat.min()
            global_max = flat.max()
            eps = 1e-8

            # Apply min–max scaling to [-1, 1]:
            #   scaled = 2 * (x - global_min) / (global_max - global_min + eps) - 1
            output = 2 * (output - global_min) / (global_max - global_min + eps) - 1

        # Squeeze out batch dim and move to CPU NumPy
        result = output.squeeze(0).cpu().numpy()  # shape: (N_maps, H, W)

    return result

def get_ref_map_(img, patch_size, n_max=10,device=None):
    kernels, weights = compute_kernels_weights(n_max=n_max, size=patch_size)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # numpy array to tensor
    torch_kernels = torch.from_numpy(kernels).float().to(device)
    torch_weights = torch.from_numpy(weights).float().to(device)

    torch_img = torch.from_numpy(img).float().to(device)

    refmap = RefMap(torch_kernels, torch_weights).to(device)

    output = refmap(torch_img)

    return output[0, :, :].cpu().numpy()

def get_ref_map(
        img: np.ndarray,
        patch_size: int,
        n_max: int = 10,
        device: Optional[torch.device] = None,
        normalize_output: bool = False,
        return_angle=False,
) -> np.ndarray:
    """
    Compute a reference map for a single 2D image and (optionally) normalize the
    output to the range [-1, 1].

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image, shape (H, W).
    patch_size : int
        Patch size to pass to `compute_kernels_weights`.
    n_max : int, optional
        Maximum weight order (default: 10).
    device : torch.device or None, optional
        If None, automatically chooses CUDA if available, else CPU.
    normalize_output : bool, optional
        If True, scale the output so its minimum becomes -1 and its maximum becomes +1.

    Returns
    -------
    np.ndarray
        A reference map of shape (H, W). If `normalize_output=True`, values lie in [-1, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Compute kernels & weights (assumed to return NumPy arrays)
    kernels, weights = compute_kernels_weights(n_max=n_max, size=patch_size)

    # 2) Move kernels & weights to device as float32 tensors
    kernels_t = torch.as_tensor(kernels, dtype=torch.float32, device=device)
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=device)

    # 3) Convert the 2D img to a tensor of shape (1, 1, H, W)
    if img.ndim != 2:
        raise ValueError(f"Expected img.ndim == 2, but got {img.ndim}")
    img_t = torch.as_tensor(img, dtype=torch.float32, device=device)

    # 4) Build the RefMap module and switch to eval mode
    refmap_model = RefMap(kernels_t, weights_t).to(device)
    refmap_model.eval()

    # 5) Forward pass under no_grad
    with torch.no_grad():
        # Output shape is assumed to be (1, 1, H, W)
        if return_angle:
            output, theta_map = refmap_model(img_t, return_angle=return_angle)
            theta_map = np.squeeze(theta_map)
        else:
            output = refmap_model(img_t, return_angle=return_angle)
            theta_map = None

        if normalize_output:
            # Flatten over all values to compute a single global min and max
            flat = output.view(-1)
            global_min = flat.min()
            global_max = flat.max()
            eps = 1e-8

            # Apply min–max scaling to [-1, 1]:
            #   scaled = 2 * (x - global_min) / (global_max - global_min + eps) - 1
            output = 2 * (output - global_min) / (global_max - global_min + eps) - 1

        # Remove batch and channel dimensions → (H, W)
        ref_map = output.squeeze(0).squeeze(0).cpu().numpy()

    return ref_map, theta_map

