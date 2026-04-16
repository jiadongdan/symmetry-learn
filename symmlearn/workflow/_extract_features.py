import numpy as np
from collections import namedtuple
from ..sampling import stratified_sampling


ExtractionResult = namedtuple('ExtractionResult', ['pts', 'patches', 'lbs'])


def extract_exp_data(img, patch_size, num_patches, lbs_img=None, seed=None):
    """
    Sample patches from an image and optionally assign dominant labels.

    Parameters
    ----------
    img : np.ndarray
        Source image of shape (H, W) or (H, W, C).
    patch_size : int
        Side length of each square patch in pixels.
    num_patches : int
        Number of patches to extract.
    lbs_img : np.ndarray, optional
        Integer label image of shape (H, W). When provided, the dominant
        (most-frequent) label within each patch region is returned.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ExtractionResult
        Named tuple with fields:
        - pts     : np.ndarray, shape (n, 2), (x, y) coordinates
        - patches : np.ndarray, shape (n, patch_size, patch_size[, C])
        - lbs     : np.ndarray of shape (n,) with dominant labels, or None
    """
    h, w = img.shape[:2]
    half = patch_size // 2

    # Generate candidate points via stratified sampling; returned as (x, y)
    pts = stratified_sampling(size=(h, w), n_samples=num_patches, seed=seed)
    pts = np.round(pts).astype(np.int64)

    # Keep only points whose patch stays fully within image bounds
    valid = (
        (pts[:, 0] >= half) &
        (pts[:, 0] + half < w) &
        (pts[:, 1] >= half) &
        (pts[:, 1] + half < h)
    )
    pts = pts[valid]

    # Down-select to num_patches when sampling yields more than needed
    if len(pts) > num_patches:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=num_patches, replace=False)
        pts = pts[idx]

    # Extract patches; pts are (x, y), numpy arrays are indexed [y, x]
    patches = np.array([
        img[y - half:y + half, x - half:x + half]
        for x, y in pts
    ])

    # Compute dominant label per patch, or return None
    if lbs_img is None:
        lbs = None
    else:
        lbs = []
        for x, y in pts:
            region = lbs_img[y - half:y + half, x - half:x + half].ravel()
            values, counts = np.unique(region, return_counts=True)
            lbs.append(values[np.argmax(counts)])
        lbs = np.array(lbs)

    return ExtractionResult(pts=pts, patches=patches, lbs=lbs)
