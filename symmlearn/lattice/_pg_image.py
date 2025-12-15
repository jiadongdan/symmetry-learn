import numbers
import numpy as np

from ase import Atoms
from scipy.spatial import Delaunay
from itertools import combinations

from ..maps import get_rot_maps, get_ref_map
from ._tapered_gaussian import add_tapered_gaussian

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
    elif method == 'max':
        stat = dists.max()
    else:
        raise ValueError(f"Invalid method '{method}'; choose from 'min', 'max', 'mean', 'median'")

    return stat

def atoms2image(atoms, size=512, sigma_map=None, amplitude_map=None, tol=1e-6, shift_range=0.0):
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
        sigma_val = _estimate_sigma(atoms, method='mean') * (size)
        sigma_map = {s: sigma_val for s in unique_syms}
    elif isinstance(sigma_map, numbers.Number):
        # single float → broadcast to all
        sigma_map = {s: float(sigma_map) for s in unique_syms}
    else:
        # assume dict; you might validate keys here if desired
        sigma_map = {s: sigma_map[s] for s in unique_syms}

    img = np.zeros((size, size), float)
    for s in unique_syms:
        pos = atoms.get_scaled_positions()[np.array(atoms.get_chemical_symbols()) == s][:, 0:2] * size
        add_tapered_gaussian(img, pts=pos, sigma=sigma_map[s], amplitude=amplitude_map[s], shift_range=shift_range)

    return img

def estimate_patch_size(unit_cell, scale=2.0):
    """
    Estimate minimum patch size to fully contain a unit cell.

    Computes the diameter of the bounding circle by finding the length
    of the longest diagonal of the unit cell parallelogram.

    Parameters
    ----------
    unit_cell : ase.Cell
        ASE Cell object representing the crystallographic unit cell.
    scale : float, optional
        Scaling factor for the patch size. Default is 2.0 to ensure the
        patch diameter equals the longest diagonal. Values > 2.0 provide
        additional margin around the unit cell.

    Returns
    -------
    float
        Estimated patch size (scaled diameter of bounding circle) in the
        same units as the unit cell parameters.
    """
    #cell = unit_cell.get_cell()
    a_vec = unit_cell[0, 0:2]
    b_vec = unit_cell[1, 0:2]

    # The two diagonals of the parallelogram
    diag1 = np.linalg.norm(a_vec + b_vec)
    diag2 = np.linalg.norm(a_vec - b_vec)

    # Diameter is the longer diagonal scaled by the scaling factor
    return scale * max(diag1, diag2)


class PGLattice:

    def __init__(self, pg_number, atoms, unit_cell_atoms, sigma_method='mean'):
        self.pg_number = pg_number
        self.atoms = atoms
        self.unit_cell_atoms = unit_cell_atoms
        self.unit_cell = unit_cell_atoms.get_cell()
        self.sigma_ = _estimate_sigma(self.atoms, method=sigma_method)
        self.size = int(self.atoms.get_cell().cellpar()[0])

    def get_image(self, image_size=None, sigma_map=None, amplitude_map=None, seed=None, shift_range=0.0):
        if image_size is None:
            image_size = self.size
        rng = np.random.default_rng(seed)
        if sigma_map is None:
            sigma_min = self.sigma_ * (image_size) * 0.16
            sigma_max = self.sigma_ * (image_size) * 0.357
            sigma_map = rng.uniform(sigma_min, sigma_max)
            sigma_map = max(1.0, sigma_map)


        img = atoms2image(self.atoms,
                          size=image_size,
                          sigma_map=sigma_map,
                          amplitude_map=amplitude_map,
                          shift_range=shift_range,
                          )

        # estimate patch size from unit cell, s is float number
        s = estimate_patch_size(self.unit_cell)
        # make the patch size odd number
        patch_size = int(s//2 * 2 + 1)
        return PGImage(self.pg_number, img, patch_size)


class PGImage:

    def __init__(self, pg_number, img, patch_size):
        self.pg_number = pg_number
        self.img = img
        self.patch_size = patch_size
        s = int(self.patch_size // 2)
        self.img_crop = self.img[s:-s, s:-s]
        self.rot_maps = None
        self.ref_map = None
        self.theta_map = None
        self.sin_map = None
        self.cos_map = None
        self.has_symm_maps = False

        self.ps = None


    def compute_symm_maps(self,
                          n_max=12,
                          patch_size=None,
                          normalize_rot=False,
                          return_angle=True,
                          p=2,
                          crop=False,
                          ):
        if patch_size is None:
            patch_size = self.patch_size
        # get the rotational and reflectional maps
        rot_maps = get_rot_maps(
            self.img,
            n_max=n_max,
            patch_size=patch_size,
            normalize_output=normalize_rot
        )
        ref_map, theta_map = get_ref_map(
            self.img,
            n_max=n_max,
            patch_size=patch_size,
            return_angle=return_angle,
            p=p
        )
        # crop
        if crop:
            s = self.patch_size // 2
            self.rot_maps = rot_maps[:, s:-s, s:-s]
            self.ref_map = ref_map[s:-s, s:-s]
            self.theta_map = theta_map[s:-s, s:-s]
        else:
            self.rot_maps = rot_maps
            self.ref_map = ref_map
            self.theta_map = theta_map

        self.sin_map = np.sin(self.theta_map * 2)
        self.cos_map = np.cos(self.theta_map * 2)

        self.has_symm_maps = True

