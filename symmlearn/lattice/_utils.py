import numpy as np
from collections import defaultdict
from typing import List

# Mapping from layer‐group number (1–80) to wallpaper (plane)‐group number (1–17)
_lg2pg = {
    1:1,  2:2,  3:2,  4:1,  5:1,  6:2,  7:2,  8:3,  9:4, 10:5,
    11:3, 12:4, 13:5, 14:6, 15:7, 16:7, 17:8, 18:9, 19:6, 20:7,
    21:8, 22:9, 23:6, 24:7, 25:8, 26:9, 27:3, 28:3, 29:4, 30:3,
    31:3, 32:4, 33:4, 34:5, 35:5, 36:3, 37:6, 38:6, 39:10,40:7,
    41:6, 42:9, 43:7, 44:8, 45:7, 46:10,47:9, 48:6, 49:10,50:10,
    51:10,52:12,53:11,54:12,55:11,56:12,57:11,58:12,59:11,60:12,
    61:11,62:11,63:12,64:11,65:13,66:16,67:14,68:15,69:14,70:15,
    71:17,72:17,73:16,74:13,75:16,76:17,77:17,78:14,79:15,80:17
}

# one-to-one map from plane-group → layer-group
_plane_to_layer = {
    1:  1,   2:  3,   3: 11,   4:  9,   5: 10,
    6: 23,   7: 24,   8: 25,   9: 26,  10: 49,
    11: 55,  12: 52,  13: 65,  14: 69,  15: 70,
    16: 73,  17: 77,
}

# Precompute the inverse mapping: wallpaper‐group → list of layer‐groups
_pg2lg: dict[int, List[int]] = defaultdict(list)
for lg, pg in _lg2pg.items():
    _pg2lg[pg].append(lg)


def layer2plane(layer_group: int) -> int:
    """
    Convert a layer‐group number (1–80) to its corresponding
    wallpaper (plane)‐group number (1–17).

    Raises KeyError if layer_group is not in 1..80.
    """
    if layer_group not in _lg2pg:
        raise KeyError(f"Invalid layer‐group number: {layer_group}")
    return _lg2pg[layer_group]

def plane2layer(plane_group: int) -> List[int]:
    """
    Convert a wallpaper (plane)‐group number (1–17) to the list
    of all layer‐group numbers that map to it.

    Raises KeyError if plane_group is not in 1..17.
    """
    if plane_group not in _pg2lg:
        raise KeyError(f"No layer groups found for wallpaper‐group #{plane_group}")
    return sorted(_pg2lg[plane_group])


def rotate_atoms_xy_center(atoms, angle_deg, rotate_cell=True):
    """
    Rotate an ASE Atoms object in the XY plane around the center (a/2, b/2).

    Parameters
    ----------
    atoms : ase.Atoms
        The Atoms object to be rotated. Assumes orthogonal cell with a = b.
    angle_deg : float
        The rotation angle in degrees (counterclockwise).
    """
    a = atoms.cell[0, 0]
    b = atoms.cell[1, 1]
    center = (a / 2, b / 2, 0)
    atoms.rotate('z', angle_deg, center=center, rotate_cell=rotate_cell)
    return atoms

def crop_atoms_xy_center(atoms, a_new=None, b_new=None):
    """
    Crop atoms from the center of the cell in the XY plane and update the cell.

    Parameters
    ----------
    atoms : ase.Atoms
        The original Atoms object.
    a_new : float, optional
        New width in x-direction. Default is a / sqrt(2).
    b_new : float, optional
        New height in y-direction. Default is b / sqrt(2).

    Returns
    -------
    ase.Atoms
        A new Atoms object cropped from the center, with updated cell.
    """
    a = atoms.cell[0, 0]
    b = atoms.cell[1, 1]
    c = atoms.cell[2, 2]

    if a_new is None:
        a_new = a / np.sqrt(2)
    if b_new is None:
        b_new = b / np.sqrt(2)

    center = np.array([a / 2, b / 2])
    lower = center - np.array([a_new / 2, b_new / 2])
    upper = center + np.array([a_new / 2, b_new / 2])

    # Filter atoms in the new region
    positions = atoms.get_positions()
    in_crop = ((positions[:, 0] >= lower[0]) & (positions[:, 0] <= upper[0]) &
               (positions[:, 1] >= lower[1]) & (positions[:, 1] <= upper[1]))

    cropped = atoms[in_crop].copy()

    # Shift positions so new cell starts at (0, 0, 0)
    cropped.positions -= np.array([lower[0], lower[1], 0])

    # Update cell
    new_cell = atoms.cell.copy()
    new_cell[0, 0] = a_new
    new_cell[1, 1] = b_new
    cropped.set_cell(new_cell)
    cropped.set_pbc(atoms.get_pbc())

    return cropped


def make_cell_square(atoms):
    """
    Extract atoms within the largest inscribed square of the cell.

    The inscribed square is found by:
    1. Finding the inscribed circle (largest circle fitting in the cell)
    2. Getting the inscribed square within that circle

    Parameters
    ----------
    atoms : ase.Atoms
        Input atoms object

    Returns
    -------
    ase.Atoms
        New atoms object with only atoms within the inscribed square
    """
    import numpy as np
    from ase import Atoms

    cell = atoms.get_cell()

    # Get the 2D cell vectors (assuming we care about xy plane)
    a = cell[0, :2]
    b = cell[1, :2]

    # Find the inscribed circle radius
    # For a parallelogram, the inscribed circle radius is:
    # r = area / semi-perimeter = |a × b| / (|a| + |b|)
    # But more precisely, it's the minimum distance from center to any edge

    # Calculate distances from origin to each edge of the parallelogram
    # The four edges are: along a, along b, along a+b from b, along a+b from a

    len_a = np.linalg.norm(a)
    len_b = np.linalg.norm(b)

    # Height of parallelogram (perpendicular distance between parallel sides)
    cross = abs(a[0] * b[1] - a[1] * b[0])  # 2D cross product (area)
    h_a = cross / len_a  # distance between edges parallel to a
    h_b = cross / len_b  # distance between edges parallel to b

    # The inscribed circle radius is half the minimum height
    r_inscribed = min(h_a, h_b) / 2

    # The inscribed square within a circle has side length s = r * sqrt(2)
    # (diagonal of square = diameter of circle = 2r, so s = 2r/sqrt(2) = r*sqrt(2))
    square_side = r_inscribed * np.sqrt(2)
    half_side = square_side / 2

    # Center of the cell
    center = (a + b) / 2

    # Get atom positions in 2D
    positions = atoms.get_positions()
    pos_2d = positions[:, :2]

    # Find atoms within the square centered at cell center
    # Square is axis-aligned
    relative_pos = pos_2d - center
    within_square = (np.abs(relative_pos[:, 0]) <= half_side) & \
                    (np.abs(relative_pos[:, 1]) <= half_side)

    # Create new atoms object with filtered atoms
    new_atoms = Atoms(
        symbols=[atoms[i].symbol for i in range(len(atoms)) if within_square[i]],
        positions=positions[within_square],
        cell=[square_side, square_side, cell[2, 2]],
        pbc=atoms.pbc
    )

    # Center the atoms in the new cell
    new_positions = new_atoms.get_positions()
    new_positions[:, 0] -= (center[0] - half_side)
    new_positions[:, 1] -= (center[1] - half_side)
    new_atoms.set_positions(new_positions)

    return new_atoms

def rescale_square_atoms(atoms, size):
    """
    Rescale a square cell to a new size.

    Parameters
    ----------
    atoms : ase.Atoms
        Input atoms object with a square cell
    size : float
        New cell side length

    Returns
    -------
    ase.Atoms
        Rescaled atoms object

    Raises
    ------
    ValueError
        If cell is not square
    """
    cell = atoms.get_cell()

    # Validate square cell
    a, b = cell[0, 0], cell[1, 1]
    if not np.isclose(a, b, rtol=1e-5):
        raise ValueError(f"Cell is not square: {a:.4f} x {b:.4f}")

    new_atoms = atoms.copy()
    new_atoms.set_cell([size, size, cell[2, 2]], scale_atoms=True)

    return new_atoms