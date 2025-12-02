import numpy as np

def add_tapered_gaussian(img, pts, sigma, amplitude=1, r_factor=3.0):
    """
    Add tapered 2D Gaussians to an image at given point locations.

    Parameters
    ----------
    img : 2D np.ndarray (float)
        Target image. This array is modified in-place and also returned.
    pts : np.ndarray, shape (N, 2)
        Floating-point coordinates of centers.
        Convention: pts[i] = (y, x) in pixel coordinates (row, col).
        Centers may be outside the image; as long as their support overlaps
        the image, they will contribute.
    sigma : float
        Standard deviation of the (isotropic) Gaussian in pixels.
    amplitude : float or array-like
        Peak amplitude of the Gaussian(s). If scalar, applied to all points.
        If array-like of length N, amplitudes per point.
    r_factor : float, optional
        Cutoff radius factor, default 3.0 → Gaussian tapered to zero at r = 3*sigma.

    Returns
    -------
    img : 2D np.ndarray
        The modified image (same object as input).
    """
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")

    h, w = img.shape
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N, 2)")

    # Handle amplitude as scalar or per-point array
    amps = np.asarray(amplitude, dtype=float)
    if amps.ndim == 0:
        amps = np.full(len(pts), float(amps), dtype=float)
    elif amps.shape[0] != len(pts):
        raise ValueError("If amplitude is array-like, its length must match number of points")

    R = r_factor * float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    sigma2 = float(sigma) ** 2

    for (y0, x0), A in zip(pts, amps):
        # Local integer bounds for patch around this point, BEFORE clipping
        y_min = int(np.floor(y0 - R))
        y_max = int(np.ceil(y0 + R)) + 1  # +1 so the upper index is exclusive
        x_min = int(np.floor(x0 - R))
        x_max = int(np.ceil(x0 + R)) + 1

        # Intersect with image bounds [0, h) × [0, w)
        y_min_clipped = max(y_min, 0)
        y_max_clipped = min(y_max, h)
        x_min_clipped = max(x_min, 0)
        x_max_clipped = min(x_max, w)

        # If there is no overlap between the support disk and the image, skip
        if y_min_clipped >= y_max_clipped or x_min_clipped >= x_max_clipped:
            continue

        # Local grid over the intersection region
        ys = np.arange(y_min_clipped, y_max_clipped)
        xs = np.arange(x_min_clipped, x_max_clipped)
        Y, X = np.meshgrid(ys, xs, indexing="ij")

        # Distance from the true (possibly outside) center
        dy = Y - y0
        dx = X - x0
        r = np.sqrt(dx * dx + dy * dy)

        # Mask within cutoff radius
        mask = r <= R
        if not np.any(mask):
            continue

        # Smoothstep taper: s = 1 - 3 t^2 + 2 t^3, with t = r / R in [0, 1]
        t = np.zeros_like(r, dtype=float)
        t[mask] = r[mask] / R
        s = np.zeros_like(r, dtype=float)
        s[mask] = 1.0 - 3.0 * t[mask]**2 + 2.0 * t[mask]**3

        # Base Gaussian
        g = np.zeros_like(r, dtype=float)
        g[mask] = np.exp(-0.5 * (r[mask] ** 2) / sigma2)

        # Tapered Gaussian
        w_local = A * g * s

        # Add into image (only where mask is true)
        img[y_min_clipped:y_max_clipped, x_min_clipped:x_max_clipped] += w_local

    return img

