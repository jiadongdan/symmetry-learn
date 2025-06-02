import numpy as np
import numbers
from ase import Atoms
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from itertools import combinations

from mtflearn.features import KeyPoints
from mtflearn.features import ZPs

from ..sampling._poisson_disk_sampling import poisson_disk_sampling

def _estimate_sigma(atoms, method='mean'):
    """
    Estimate a single Gaussian sigma (in pixels) from the Delaunay edge lengths
    of the atoms' scaled XY positions.

    Parameters
    ----------
    atoms : ase.Atoms
        Your structure (must have a square, orthogonal XY cell).
    size : int
        Image resolution (size × size).
    method : {'min', 'mean', 'median'}
        Which statistic of the edge-length distribution to use.

    Returns
    -------
    sigma_px : float
        Estimated sigma in pixel units.
    """
    # get scaled XY positions
    scaled = atoms.get_scaled_positions()[:, :2]  # shape (N,2)
    tri = Delaunay(scaled)

    # collect unique edges
    edges = set()
    for simplex in tri.simplices:
        for i, j in combinations(simplex, 2):
            edges.add(tuple(sorted((i, j))))

    # compute edge lengths
    dists = np.array([np.linalg.norm(scaled[i] - scaled[j]) for i, j in edges])

    # choose statistic
    method = method.lower()
    if method == 'min':
        stat = dists.min()
    elif method == 'median':
        stat = np.median(dists)
    elif method == 'mean':
        stat = dists.mean()
    else:
        raise ValueError(f"Invalid method '{method}'; choose from 'min', 'mean', 'median'")

    return stat/4.


def atoms2image(atoms, size=512, sigma_map=None, amplitude_map=None, tol=1e-6):
    """
    Convert an ASE Atoms into a 2D image by dropping impulses at each atom's
    fractional (x,y) positions, blurring per element, summing, and normalizing.

    Parameters
    ----------
    atoms : ase.Atoms
        Must have an orthogonal, square cell in the XY plane.
    size : int
        Output resolution (size × size).
    sigma_map : None, float, or dict[str, float], optional
        - None: auto-estimate a single sigma for all elements.
        - float: use that sigma for every element.
        - dict: map each element symbol → its sigma in pixels.
    amplitude_map : dict[str, float], optional
        Impulse amplitude per element; defaults to 1.0 for all.
    tol : float
        Tolerance for cell‐orthogonality and equality checks.
    """
    # 1) Check cell is square & orthogonal in XY
    cell = atoms.get_cell()
    a_vec, b_vec = cell[0], cell[1]
    a_len, b_len = np.linalg.norm(a_vec), np.linalg.norm(b_vec)
    if abs(np.dot(a_vec, b_vec)) > tol:
        raise ValueError("Cell a·b ≠ 0 → not orthogonal")
    if abs(a_len - b_len) > tol:
        raise ValueError("Cell sides differ → not square")

    # 2) Symbols & default amplitudes
    symbols    = atoms.get_chemical_symbols()
    unique_syms = sorted(set(symbols))
    if amplitude_map is None:
        amplitude_map = {s: 1.0 for s in unique_syms}

    # 3) Build sigma_map dict
    if sigma_map is None:
        # estimate one sigma and apply to all
        sigma_val = _estimate_sigma(atoms, size)
        sigma_map = {s: sigma_val for s in unique_syms}
    elif isinstance(sigma_map, numbers.Number):
        # single float → broadcast to all
        sigma_map = {s: float(sigma_map) for s in unique_syms}
    else:
        # assume dict; you might validate keys here if desired
        sigma_map = {s: float(sigma_map.get(s, _estimate_sigma(atoms, size)))
                     for s in unique_syms}

    # 4) Deposit impulses at fractional coords
    images = {s: np.zeros((size, size), float) for s in unique_syms}
    scaled = atoms.get_scaled_positions()
    for (x_f, y_f, _), s in zip(scaled, symbols):
        i = int(np.clip(x_f * (size - 1), 0, size - 1))
        j = int(np.clip(y_f * (size - 1), 0, size - 1))
        images[s][j, i] += amplitude_map[s]

    # 5) Blur each channel & sum
    img = np.zeros((size, size), float)
    for s, im in images.items():
        img += gaussian_filter(im, sigma=sigma_map[s], mode='constant')

    # 6) Normalize to [0,1]
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img.fill(0.0)

    return img

def estimate_patch_size(atoms, unit_cell, image_size):
    """
    Estimate how many pixels (patch size) correspond to one unit cell,
    given a full‐image that covers `atoms` on an `image_size`×`image_size` grid.
    The result is rounded to the nearest odd integer.

    Parameters
    ----------
    atoms : ase.Atoms
        The larger system whose full‐cell image is `image_size` px on a side.
    unit_cell : array-like, shape (3,3)
        The 3×3 cell matrix of the smaller “unit” cell whose patch size
        you want to extract (in the same length units as `atoms.get_cell()`).
    image_size : int or float
        The pixel width (and height) of the square image of `atoms`.

    Returns
    -------
    int
        Side length in pixels (odd integer) of one `unit_cell` patch.
    """
    def _xy_square_side(cell):
        # project the a- and b-vectors onto xy and get the bounding-box side
        c = np.asarray(cell)
        v1, v2 = c[0][:2], c[1][:2]
        corners = np.array([[0, 0], v1, v2, v1 + v2])
        mins, maxs = corners.min(axis=0), corners.max(axis=0)
        w, h = maxs - mins
        return float(max(w, h))

    full_side = _xy_square_side(atoms.get_cell())
    unit_side = _xy_square_side(unit_cell)
    raw       = (unit_side / full_side) * image_size * 2

    n = int(round(raw))
    if n % 2 == 1:
        return n

    # choose the nearest odd neighbor
    lower, higher = n - 1, n + 1
    if lower < 1:
        return higher
    return lower if abs(raw - lower) <= abs(raw - higher) else higher

class PGLattice:

    def __init__(self, pg_number, atoms, unit_cell):
        self.pg_number = pg_number
        self.atoms = atoms
        self.unit_cell = unit_cell
        self.sigma_ = _estimate_sigma(self.atoms)

    def get_image(self, size=512, sigma_map=None, amplitude_map=None):
        if sigma_map is None:
            sigma_map = self.sigma_ * (size - 1)
        # estimate patch size
        patch_size = estimate_patch_size(self.atoms, self.unit_cell, size)
        img = atoms2image(self.atoms, size=size, sigma_map=sigma_map, amplitude_map=amplitude_map)
        return PGImage(self.pg_number, img, patch_size)


class PGImage:

    def __init__(self, pg_number, data, patch_size):
        self.pg_number = pg_number
        self.data = data
        self.patch_size = patch_size
        self.symmetry_maps = None


    def get_X(self, n_max=10, radius=None, seed=None):
        if radius is None:
            radius = self.patch_size / 3.
        # get the points
        self.pts = poisson_disk_sampling(radius=radius, size=self.data.shape[0], seed=seed)
        # extract patches
        kp = KeyPoints(self.pts, self.data, self.patch_size)
        self.ps = kp.extract_patches(self.patch_size)
        self.pts = kp.pts
        # get features
        zps = ZPs(n_max=n_max, size=self.patch_size)
        m = zps.fit_transform(self.ps)
        return m