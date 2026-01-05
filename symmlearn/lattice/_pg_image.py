import numbers
import numpy as np

from ase import Atoms
from scipy.spatial import Delaunay
from itertools import combinations
from scipy.spatial.distance import cdist

from ..maps import get_rot_maps, get_ref_map
from ._tapered_gaussian import add_tapered_gaussian
from ._mixin_plane_group import transform_via_cell
from ._mixin_plane_group import PG_PATTERNS

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

def coords_to_ase_cell(coords, z_height=12.0, pbc=None):

    coords = np.array(coords)

    if pbc is None:
        pbc = [True, True, False]

    # Take two adjacent sides as cell vectors
    a = coords[1] - coords[0]  # First lattice vector
    b = coords[2] - coords[0]  # Second lattice vector

    # Create 3x3 cell matrix (add z-dimension)
    cell = np.array([
        [a[0], a[1], 0.0],
        [b[0], b[1], 0.0],
        [0.0, 0.0, z_height]
    ])

    return cell

def get_line(p1, p2):
    """
    Get integer positions of line points connecting p1 and p2.
    Uses Bresenham's line algorithm for efficient integer line drawing.

    Parameters
    ----------
    p1 : array-like, shape (2,)
        Starting point (x, y) in float
    p2 : array-like, shape (2,)
        Ending point (x, y) in float

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        Array of integer (x, y) coordinates along the line
    """
    # Convert to integers
    x1, y1 = int(round(p1[0])), int(round(p1[1]))
    x2, y2 = int(round(p2[0])), int(round(p2[1]))

    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Determine direction of line
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    err = dx - dy

    x, y = x1, y1

    while True:
        points.append([x, y])

        # Reached endpoint
        if x == x2 and y == y2:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x += sx

        if e2 < dx:
            err += dx
            y += sy

    return np.array(points)

def find_translation_vector(
        large_atoms: Atoms,
        small_atoms: Atoms,
        tolerance: float = 1e-3,
        z_fix: bool = True
) -> np.ndarray:
    """
    Find the 2D translation vector from small_atoms to its matching position in large_atoms.

    Searches for the small atomic structure within the larger one by trying to match
    atomic positions using nearest neighbor matching in the xy-plane. The returned
    translation vector is adjusted to point near the center of the large_atoms cell.

    Args:
        large_atoms: The larger ASE Atoms object containing the small structure.
        small_atoms: The smaller ASE Atoms object to locate within large_atoms.
        tolerance: Maximum distance (Å) for considering positions matched. Default: 1e-3.
        z_fix: If True, sets z-component of translation to 0. Default: True.

    Returns:
        Translation vector [tx, ty] to align small_atoms with large_atoms, adjusted
        to point near the center of the large cell using periodic boundary conditions.

    Raises:
        ValueError: If no matching substructure found within tolerance, or if
                   small_atoms contains more atoms than large_atoms.
    """
    large_pos = large_atoms.get_positions()
    small_pos = small_atoms.get_positions()

    if len(small_pos) > len(large_pos):
        raise ValueError(
            "Small atoms object cannot contain more atoms than large atoms object."
        )

    atomic_numbers_large = large_atoms.get_atomic_numbers()
    atomic_numbers_small = small_atoms.get_atomic_numbers()

    # Extract lattice vectors for periodic boundary adjustment
    small_a_vec = small_atoms.cell[0, :2]
    small_b_vec = small_atoms.cell[1, :2]

    # Center of the large cell (in 2D)
    large_center = np.sum(large_atoms.cell[:2, :2], axis=0) / 2

    # Try each atom in large structure as potential alignment anchor
    for i_anchor in range(len(large_pos)):
        # Skip if atomic species doesn't match first atom in small structure
        if atomic_numbers_large[i_anchor] != atomic_numbers_small[0]:
            continue

        # Calculate translation needed to align first small atom with current anchor
        translation_xy = large_pos[i_anchor, :2] - small_pos[0, :2]
        translation_z = 0.0 if z_fix else (large_pos[i_anchor, 2] - small_pos[0, 2])

        # Apply translation to small structure
        translated_pos = small_pos.copy()
        translated_pos[:, :2] += translation_xy
        if not z_fix:
            translated_pos[:, 2] += translation_z

        # Check if translation produces valid match
        if _is_valid_match(
                translated_pos,
                large_pos,
                atomic_numbers_small,
                atomic_numbers_large,
                tolerance
        ):
            # Adjust translation to point near the center using PBC
            adjusted_translation = _center_translation_vector(
                translation_xy,
                large_center,
                small_a_vec,
                small_b_vec
            )

            return adjusted_translation

    raise ValueError(
        f"Could not find matching substructure within tolerance {tolerance} Å."
    )


def _center_translation_vector(
        translation: np.ndarray,
        target_center: np.ndarray,
        a_vec: np.ndarray,
        b_vec: np.ndarray
) -> np.ndarray:
    """
    Adjust translation vector to point near a target center using periodic images.

    This function finds the optimal integer linear combination of lattice vectors
    to add to the translation so that it points as close as possible to the target.

    Args:
        translation: Initial translation vector [tx, ty].
        target_center: Target position to point towards [cx, cy].
        a_vec: First lattice vector (2D).
        b_vec: Second lattice vector (2D).

    Returns:
        Adjusted translation vector closest to target_center.
    """
    # We want to minimize |translation + n*a_vec + m*b_vec - target_center|
    # This is equivalent to finding (n, m) that brings translation closest to target_center

    # Calculate the displacement we want to add
    displacement = target_center - translation

    # Solve for (n, m) using least squares: [a_vec, b_vec] * [n, m]^T = displacement
    # Stack lattice vectors as columns
    lattice_matrix = np.column_stack([a_vec, b_vec])

    # Solve for coefficients (allowing fractional values)
    coeffs, _, _, _ = np.linalg.lstsq(lattice_matrix, displacement, rcond=None)

    # Round to nearest integers
    n, m = np.round(coeffs).astype(int)

    # Apply the shift
    adjusted_translation = translation + n * a_vec + m * b_vec

    return adjusted_translation


def _is_valid_match(
        translated_pos: np.ndarray,
        large_pos: np.ndarray,
        atomic_numbers_small: np.ndarray,
        atomic_numbers_large: np.ndarray,
        tolerance: float
) -> bool:
    """
    Check if translated small structure matches a subregion of large structure.

    Args:
        translated_pos: Positions of small structure after translation.
        large_pos: Positions of all atoms in large structure.
        atomic_numbers_small: Atomic numbers of small structure.
        atomic_numbers_large: Atomic numbers of large structure.
        tolerance: Maximum distance for position matching.

    Returns:
        True if all atoms match within tolerance with correct species.
    """
    # Find nearest neighbor in large structure for each translated atom
    dist_matrix = cdist(translated_pos, large_pos, metric='euclidean')
    min_distances = np.min(dist_matrix, axis=1)
    closest_indices = np.argmin(dist_matrix, axis=1)

    # Check distance threshold
    if not np.all(min_distances < tolerance):
        return False

    # Check atomic species match
    matched_atomic_numbers = atomic_numbers_large[closest_indices]
    return np.array_equal(atomic_numbers_small, matched_atomic_numbers)

class PGLattice:

    def __init__(self, pg_number, atoms, unit_cell_atoms, sigma_method='mean'):
        self.pg_number = pg_number
        self.atoms = atoms
        self.unit_cell_atoms = unit_cell_atoms
        self.unit_cell = unit_cell_atoms.get_cell()
        self.sigma_ = _estimate_sigma(self.atoms, method=sigma_method)
        self.size = int(self.atoms.get_cell().cellpar()[0])

    def get_image(self, image_size, sigma_map=None, amplitude_map=None, seed=None, shift_range=0.0):
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

        # Crop unit cell image from the supercell image
        # Get unit cell dimensions in pixels
        uc_cell = self.unit_cell_atoms.get_cell()
        uc_len = np.linalg.norm(uc_cell[0, 0:2])  # Unit cell size in physical units
        supercell_len = self.atoms.get_cell().cellpar()[0]  # Supercell size in physical units

        # Calculate scale factor from physical units to pixels
        supercell_cell = self.atoms.get_cell()
        supercell_a = np.linalg.norm(supercell_cell[0, 0:2])
        scale = image_size / supercell_a  # pixels per physical unit

        # Get unit cell vectors in pixel coordinates
        uc_cell = self.unit_cell_atoms.get_cell()
        uc_a_vec = uc_cell[0, 0:2] * scale  # unit cell a vector in pixels
        uc_b_vec = uc_cell[1, 0:2] * scale  # unit cell b vector in pixels

        #img_copy = img.copy()
        #n = img.shape[0]//4  # border thickness

        # Fill n-pixel thick border with zeros
        #img_copy[:n, :] = 0      # top n rows
        #img_copy[-n:, :] = 0     # bottom n rows
        #img_copy[:, :n] = 0      # left n columns
        #img_copy[:, -n:] = 0     # right n columns

        #row, col = np.unravel_index(np.argmax(img_copy), img_copy.shape)
        # uc_origin_px = np.array([col, row])
        # uc_origin_px = np.array([0, 0])

        uc_origin_px = find_translation_vector(self.atoms, self.unit_cell_atoms)

        # Calculate four corners of the unit cell parallelogram
        corner_0 = uc_origin_px  # Origin
        corner_1 = uc_origin_px + uc_a_vec  # Along a
        corner_2 = uc_origin_px + uc_b_vec  # Along b
        corner_3 = uc_origin_px + uc_a_vec + uc_b_vec  # Opposite corner

        unit_cell_corners = np.array([corner_0, corner_1, corner_2, corner_3])

        return PGImage(self.pg_number, img, patch_size, unit_cell_corners)


class PGImage:

    def __init__(self, pg_number, img, patch_size, unit_cell_corners):
        self.pg_number = pg_number
        self.unit_cell_corners = unit_cell_corners
        self.unit_cell = coords_to_ase_cell(unit_cell_corners)
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

    def get_rot_centers(self, n_fold=3):
        try:
            pattern = PG_PATTERNS[self.pg_number]
        except KeyError:
            raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")
        # transform all coordinate sets through the cell
        P2 = transform_via_cell(pattern.P2, self.unit_cell)
        P3 = transform_via_cell(pattern.P3, self.unit_cell)
        P4 = transform_via_cell(pattern.P4, self.unit_cell)
        P6 = transform_via_cell(pattern.P6, self.unit_cell)

        if P2 is not None:
            P2 = P2 + self.unit_cell_corners[0]
        if P3 is not None:
            P3 = P3 + self.unit_cell_corners[0]
        if P4 is not None:
            P4 = P4 + self.unit_cell_corners[0]
        if P6 is not None:
            P6 = P6 + self.unit_cell_corners[0]

        if n_fold is None:
            return (P2, P3, P4, P6)
        elif n_fold == 2:
            return P2
        elif n_fold == 3:
            return P3
        elif n_fold == 4:
            return P4
        elif n_fold == 6:
            return P6
        else:
            raise ValueError(f"n_fold must be 2, 3, 4, 6 and None")

    def get_mirror_lines(self):
        try:
            pattern = PG_PATTERNS[self.pg_number]
        except KeyError:
            raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

        mirror_pairs  = transform_via_cell(pattern.mirror_pairs, self.unit_cell)
        lines = []
        for (p1, p2) in mirror_pairs:
            line = get_line(p1, p2) + self.unit_cell_corners[0]
            lines.append(line)
        return np.vstack(lines)


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

