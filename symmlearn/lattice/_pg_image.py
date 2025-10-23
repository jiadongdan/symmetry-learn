import numbers
import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from itertools import combinations

from mtflearn.features import KeyPoints
from mtflearn.features import ZPs

from ..sampling._poisson_disk_sampling import poisson_disk_sampling
from ..maps import get_rot_maps, get_ref_map
from ._estimate_patch_size import estimate_patch_size_from_img


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

    return stat/5.   # we divide it by 5 when using mean method


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
        sigma_val = _estimate_sigma(atoms, method='mean') * (size)
        sigma_map = {s: sigma_val for s in unique_syms}
    elif isinstance(sigma_map, numbers.Number):
        # single float → broadcast to all
        sigma_map = {s: float(sigma_map) for s in unique_syms}
    else:
        # assume dict; you might validate keys here if desired
        sigma_map = {s: sigma_map[s] for s in unique_syms}

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

    def __init__(self, pg_number, atoms, unit_cell, sigma_method='min'):
        self.pg_number = pg_number
        self.atoms = atoms
        self.unit_cell = unit_cell
        self.sigma_ = _estimate_sigma(self.atoms, method=sigma_method)

    def get_image(self, size=512, sigma_map=None, amplitude_map=None):
        if sigma_map is None:
            sigma_min = 1.2
            if self.sigma_ > sigma_min:
                sigma_max = self.sigma_ * (size)
            else:
                sigma_max = 2
            sigma_map = np.random.uniform(sigma_min, sigma_max)

        # estimate patch size from atoms
        # patch_size = estimate_patch_size(self.atoms, self.unit_cell, size)
        img = atoms2image(self.atoms, size=size, sigma_map=sigma_map, amplitude_map=amplitude_map)
        # estimate patch size from img
        s = estimate_patch_size_from_img(img) * 4
        patch_size = s//2 * 2 + 1
        return PGImage(self.pg_number, img, patch_size)


class PGImage:

    def __init__(self, pg_number, img, patch_size):
        self.pg_number = pg_number
        self.img = img
        self.patch_size = patch_size
        s = self.patch_size // 2
        self.img_crop = self.img[s:-s, s:-s]
        self.rot_maps = None
        self.ref_map = None
        self.has_symm_maps = False

        self.ps = None

    @classmethod
    def from_array(cls, data, pg_number, patch_size):
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 3 or data.shape[0] < 2:
            raise ValueError("data must be (C, H, W) with C>=2: [img_crop, ref_map, rot_maps...]")

        img_crop = data[0]
        ref_map  = data[1]
        rot_maps = data[2:] if data.shape[0] > 2 else None

        # Use patch_size=1 so s=0; then immediately overwrite attributes to avoid empty slicing.
        obj = cls(pg_number=int(pg_number), img=img_crop, patch_size=patch_size)
        obj.img = img_crop
        obj.img_crop = img_crop
        obj.ref_map = ref_map
        obj.rot_maps = rot_maps
        obj.has_symm_maps = True
        return obj

    @classmethod
    def from_npz_file(cls, filename):
        pass


    def compute_symm_maps(self, n_max=12, patch_size=None, normalize_rot=True, normalize_ref=False):
        if patch_size is None:
            patch_size = self.patch_size
        # get the rotational and reflectional maps
        rot_maps = get_rot_maps(self.img, n_max=n_max, patch_size=patch_size, normalize_output=normalize_ref)
        ref_map = get_ref_map(self.img, n_max=n_max, patch_size=patch_size, normalize_output=normalize_rot)
        # crop
        s = self.patch_size // 2
        self.rot_maps = rot_maps[:, s:-s, s:-s]
        self.ref_map = ref_map[s:-s, s:-s]
        self.has_symm_maps = True

    def get_patches(self, radius=None, scale=2., seed=None):
        if radius is None:
            radius = self.patch_size / scale
        data_arrays = np.vstack([self.img_crop[np.newaxis, :, :], self.ref_map[np.newaxis, :, :], self.rot_maps])
        if self.patch_size % 2 == 0:
            s1 = self.patch_size // 2
            s2 = self.patch_size // 2
        else:
            s1 = self.patch_size // 2
            s2 = self.patch_size // 2 + 1
        # get the points
        self.pts = poisson_disk_sampling(radius=radius, size=self.img.shape[0], seed=seed)
        kp = KeyPoints(self.pts, self.img_crop, self.patch_size)
        self.pts = kp.pts
        self.ps = np.array([data_arrays[:, y-s1:y+s2, x-s1:x+s2] for (x, y) in kp.pts])
        return self.ps

    def save_pgi(self, filename):
        if not self.has_symm_maps:
            raise ValueError("Symmetry maps have not been computed. Run compute_symm_maps() first.")

        # Ensure rot_maps and ref_map are in compatible shapes
        # img_crop: (H, W)
        # ref_map:  (H, W)
        # rot_maps: (N_rot, H, W)
        # Stack into one array: shape = (1 + 1 + N_rot, H, W)
        data = np.concatenate(
            [
                self.img_crop[None, :, :],         # shape (1, H, W)
                self.ref_map[None, :, :],          # shape (1, H, W)
                self.rot_maps                      # shape (N_rot, H, W)
            ],
            axis=0
        ).astype(np.float32)  # ensure float32

        # Save to compressed NPZ
        np.savez_compressed(
            filename,
            data=data,
            pg_number=self.pg_number,
            patch_size=self.patch_size,
            ps=self.ps
        )

    def get_X(self, n_max=10, radius=None, seed=None):
        # get the rotational and reflectional maps
        rot_maps = get_rot_maps(self.img, n_max=n_max, patch_size=self.patch_size, normalize_output=True)
        ref_map = get_ref_map(self.img, n_max=n_max, patch_size=self.patch_size, normalize_output=True)
        # crop
        s = self.patch_size // 2
        self.rot_maps = rot_maps[:, s:-s, s:-s]
        self.ref_map = ref_map[s:-s, s:-s]

        if radius is None:
            radius = self.patch_size / 4.
        # get the points
        self.pts = poisson_disk_sampling(radius=radius, size=self.img.shape[0], seed=seed)
        # get ZPs
        zps = ZPs(n_max=n_max, size=self.patch_size)
        # extract patches
        data_arrays = np.vstack([self.img_crop[np.newaxis, :, :], self.ref_map[np.newaxis, :, :], self.rot_maps])
        X = []
        Xrot = []
        ps_all = []
        for data in data_arrays:
            kp = KeyPoints(self.pts, data, self.patch_size)
            ps = kp.extract_patches(self.patch_size)
            # get features
            m = zps.fit_transform(ps)
            X.append(m.data)
            Xrot.append(np.abs(m.to_complex().data))
            ps_all.append(ps)
        self.pts = kp.pts
        self.ps = np.array(ps_all)
        return np.hstack(X), np.hstack(Xrot)

    def show(self, ax=None):
        if ax is None:
            fig, axes = plt.subplots(2, 3, figsize=(8, 5))
        axes[0, 0].imshow(self.img_crop)
        axes[1, 0].imshow(self.ref_map)
        axes[0, 1].imshow(self.rot_maps[0])
        axes[0, 2].imshow(self.rot_maps[1])
        axes[1, 1].imshow(self.rot_maps[2])
        axes[1, 2].imshow(self.rot_maps[3])
        axes[0, 0].scatter(self.pts[:, 0], self.pts[:, 1], color='r', s=10)
