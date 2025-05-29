import numpy as np
from scipy.stats import qmc

def poisson_disk_sampling(
        radius,
        size=512,
        optimization='lloyd',
        seed=None,
        integer=True
):
    """
    Generate Poisson-disk sampled 2D points in a square [0, size]^2.

    Parameters
    ----------
    radius : float
        Minimum allowed distance between any two points.
    size : float, optional
        Side length of the square domain. Default is 512.
    optimization : str, optional
        ’lloyd’ or ’none’ for SciPy’s PoissonDisk.
    seed : int or None, optional
        Seed for reproducibility.
    integer : bool, optional
        If True, snap to integer grid and re-filter so that all points
        are integer coords with pairwise distance >= radius.

    Returns
    -------
    points : ndarray, shape (n_points, 2)
        The sampled 2D points (float coords if integer=False,
        integer coords if integer=True).
    """
    # 1) RNG
    rng = np.random.default_rng(seed) if seed is not None else None

    # 2) continuous Poisson disk in unit square
    sampler = qmc.PoissonDisk(d=2,
                              radius=radius/size,
                              rng=rng,
                              optimization=optimization)
    samples_unit = sampler.fill_space()

    # 3) scale up to [0, size]
    points = samples_unit * size

    if integer:
        # 4a) round to nearest integer grid
        pts = np.round(points).astype(int)
        # 4b) clip to valid box
        pts[:,0] = np.clip(pts[:,0], 0, int(size) - 1)
        pts[:,1] = np.clip(pts[:,1], 0, int(size) - 1)
        # 4c) dedupe
        pts = np.unique(pts, axis=0)

        # 4d) greedy enforce min-distance on integer grid
        accepted = []
        for p in pts:
            if all(np.linalg.norm(p - q) >= radius for q in accepted):
                accepted.append(p)
        points = np.array(accepted)

    return points
