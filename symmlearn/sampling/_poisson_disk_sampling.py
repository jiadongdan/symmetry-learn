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

# this function is faster for 2d points
def poisson_disk_2d(width, height, r, k=30, seed=None):
    """
    Fast 2D Poisson-disk sampling using Bridson's algorithm.

    Parameters
    ----------
    width : float
        Width of the rectangular domain (x in [0, width]).
    height : float
        Height of the rectangular domain (y in [0, height]).
    r : float
        Minimum allowed distance between points.
    k : int, optional
        Number of candidate samples to try for each active point (default: 30).
    seed : int or None, optional
        RNG seed for reproducibility.

    Returns
    -------
    pts : (N, 2) ndarray
        Generated points with pairwise distances >= r.
    """
    rng = np.random.default_rng(seed)

    # Cell size: so that each cell has at most one point
    cell_size = r / np.sqrt(2.0)
    grid_w = int(np.ceil(width / cell_size))
    grid_h = int(np.ceil(height / cell_size))

    # Grid to store point indices, -1 means empty
    grid = -np.ones((grid_h, grid_w), dtype=int)

    pts = []        # list of accepted points
    active = []     # list of indices of "active" points

    # Helper to add a point
    def add_point(p):
        idx = len(pts)
        pts.append(p)
        gx = int(p[0] // cell_size)
        gy = int(p[1] // cell_size)
        grid[gy, gx] = idx
        active.append(idx)

    # 1. Start with a single random point
    p0 = np.array([rng.uniform(0, width), rng.uniform(0, height)])
    add_point(p0)

    r2 = r * r

    while active:
        # Pick a random active point
        idx = rng.choice(active)
        base = pts[idx]
        found = False

        for _ in range(k):
            # Sample candidate in [r, 2r] annulus
            rad = r * (1.0 + rng.random())
            ang = rng.uniform(0.0, 2.0 * np.pi)
            offset = np.array([np.cos(ang), np.sin(ang)]) * rad
            cand = base + offset

            # Check inside domain
            if not (0 <= cand[0] <= width and 0 <= cand[1] <= height):
                continue

            # Check neighbors in grid
            gx = int(cand[0] // cell_size)
            gy = int(cand[1] // cell_size)

            ok = True
            # Neighbor cells [-2..+2] is safe (can also use [-1..+1])
            for ny in range(max(gy - 2, 0), min(gy + 3, grid_h)):
                for nx in range(max(gx - 2, 0), min(gx + 3, grid_w)):
                    j = grid[ny, nx]
                    if j != -1:
                        diff = cand - pts[j]
                        if diff.dot(diff) < r2:
                            ok = False
                            break
                if not ok:
                    break

            if ok:
                add_point(cand)
                found = True
                break

        # If no candidate was accepted, deactivate this point
        if not found:
            active.remove(idx)

    return np.array(pts, dtype=float)
