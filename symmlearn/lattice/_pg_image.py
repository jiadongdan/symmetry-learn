import numpy as np
import numbers
from ase import Atoms
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from itertools import combinations



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

    return stat


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
        img += gaussian_filter(im, sigma=sigma_map[s])

    # 6) Normalize to [0,1]
    mn, mx = img.min(), img.max()
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img.fill(0.0)

    return img

class PGLattice:

    def __init__(self, pg_number, atoms):
        self.pg_number = pg_number
        self.atoms = atoms
        self.sigma_ = _estimate_sigma(self.atoms)

    def get_image(self, size=512, sigma_map=None, amplitude_map=None):
        if sigma_map is None:
            sigma_map = self.sigma_ * (size - 1)
        return atoms2image(self.atoms, size=size, sigma_map=sigma_map, amplitude_map=amplitude_map)


class PGImage:

    def __init__(self, pg_number, data, patch_size):
        self.pg_number = pg_number
        self.data = data
        self.patch_size = patch_size

        self.symmetry_maps = None

    def get_patches(self):
        pass

    def get_X(self):
        pass