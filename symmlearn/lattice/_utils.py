def rotate_atoms_xy_center(atoms, angle_deg):
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
    atoms.rotate('z', angle_deg, center=center, rotate_cell=False)
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