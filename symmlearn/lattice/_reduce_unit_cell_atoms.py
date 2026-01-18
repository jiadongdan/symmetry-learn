import spglib
import numpy as np
from ase import Atoms
from scipy.optimize import minimize_scalar


def reduce_unit_cell_atoms(atoms):
    """
    Reduce atoms to primitive cell while preserving the original orientation.
    """
    cell_matrix = np.array(atoms.cell)
    cart_positions = atoms.get_positions()
    scaled_positions = atoms.get_scaled_positions()
    numbers = atoms.get_atomic_numbers()
    spg_cell = (cell_matrix, scaled_positions, numbers)

    prim_result = spglib.find_primitive(spg_cell)

    if prim_result is None:
        return atoms

    prim_cell_spg, prim_pos_spg, prim_nums = prim_result
    n_prim = len(prim_nums)

    if n_prim == len(numbers):
        return atoms

    def rotation_error(angle):
        """Error function: how far is M from being integer?"""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])

        rotated_prim = (R @ prim_cell_spg.T).T

        try:
            M = cell_matrix @ np.linalg.inv(rotated_prim)
            M_rounded = np.round(M)
            return np.sum((M - M_rounded) ** 2)
        except np.linalg.LinAlgError:
            return float('inf')

    # Find best angle using optimization
    # First do coarse search to find good starting point
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    errors = [rotation_error(a) for a in angles]
    best_coarse = angles[np.argmin(errors)]

    # Then refine with optimization
    result = minimize_scalar(rotation_error,
                             bounds=(best_coarse - 0.1, best_coarse + 0.1),
                             method='bounded')
    best_angle = result.x

    if rotation_error(best_angle) > 0.01:
        return Atoms(numbers=prim_nums, scaled_positions=prim_pos_spg,
                     cell=prim_cell_spg, pbc=atoms.pbc)

    # Apply best rotation
    cos_a = np.cos(best_angle)
    sin_a = np.sin(best_angle)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [0,      0,     1]
    ])
    rotated_prim_cell = (R @ prim_cell_spg.T).T

    # Extract atoms in new cell
    new_scaled = cart_positions @ np.linalg.inv(rotated_prim_cell)
    new_scaled = new_scaled % 1.0

    # Remove duplicates
    unique_positions = []
    unique_numbers = []
    tol = 1e-4

    for pos, num in zip(new_scaled, numbers):
        is_duplicate = False
        for upos in unique_positions:
            diff = pos - upos
            diff = diff - np.round(diff)
            if np.linalg.norm(diff) < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_positions.append(pos)
            unique_numbers.append(num)

    if len(unique_positions) != n_prim:
        return Atoms(numbers=prim_nums, scaled_positions=prim_pos_spg,
                     cell=prim_cell_spg, pbc=atoms.pbc)

    return Atoms(
        numbers=unique_numbers,
        scaled_positions=unique_positions,
        cell=rotated_prim_cell,
        pbc=atoms.pbc,
    )
