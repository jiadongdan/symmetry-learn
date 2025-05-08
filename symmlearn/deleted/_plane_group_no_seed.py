import re
import numpy as np
from ase.cell import Cell
from ase import Atoms
from typing import List, Tuple, Optional, Union
from symmlearn.lattice._wyckoff_position import WyckoffPosition
from symmlearn.lattice._wyckoff_position import wyckoff_pos

def generate_plane_group_cell(
        pg_number: int,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = 12,
        gamma: Optional[float] = None,
        a_range: Tuple[float, float] = (2.0, 4.0),
        b_range: Tuple[float, float] = (2.0, 4.0),
        seed: Optional[int] = None
) -> Cell:
    """
    Return an ASE Cell for a 2D wallpaper (plane) group, sampling lattice
    parameters within specified ranges if not explicitly provided.

    Args:
        pg_number: Wallpaper group number (1–17).
        a: Lattice constant along x; if None, sampled from a_range.
        b: Lattice constant along y; if None, sampled from b_range for
           oblique/rectangular groups, else set equal to a.
        c: Lattice constant along z; if None, default is 12
        gamma: Angle between a and b in degrees; if None, set or sampled by group:
            - pg 1 (oblique): random in [60,120]
            - rectangular & square (pg 2–11): 90
            - hexagonal (pg 12–17): 120
        a_range: (min, max) for sampling a when a is None.
        b_range: (min, max) for sampling b when b is None.
        seed: RNG seed for reproducible sampling.

    Returns:
        An ase.cell.Cell object with the in-plane vectors defined and
        a fixed z-axis of length 12.
    """
    rng = np.random.default_rng(seed)

    # 2D Bravais lattice classes
    oblique     = {1, 2}
    rectangular = set(range(3, 10))
    square      = {10, 11, 12}
    hexagonal   = set(range(13, 18))

    # Sample a if needed
    if a is None:
        a = float(rng.uniform(a_range[0], a_range[1]))

    # Sample or set b
    if b is None:
        if pg_number in oblique or pg_number in rectangular:
            b = float(rng.uniform(b_range[0], b_range[1]))
        else:
            b = a

    # Sample or set gamma
    if gamma is None:
        if pg_number in oblique:
            gamma = float(rng.uniform(60.0, 120.0))
        elif pg_number in rectangular or pg_number in square:
            gamma = 90.0
        elif pg_number in hexagonal:
            gamma = 120.0
        else:
            raise ValueError(f"Plane group must be 1–17; got {pg_number}")

    # Build lattice vectors
    gamma_rad = np.deg2rad(gamma)
    lattice = np.array([
        [a,                      0.0,                   0.0],
        [b * np.cos(gamma_rad),  b * np.sin(gamma_rad), 0.0],
        [0.0,                    0.0,                     c],
    ], dtype=float)

    return Cell(lattice)
def _min_dist_metric(atoms: Atoms) -> float:
    """Minimum non-zero interatomic distance under PBC in x,y."""
    dmat = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dmat, np.inf)
    return float(dmat.min())

def _avg_nn_metric(atoms: Atoms) -> float:
    """Average nearest-neighbor distance under PBC in x,y."""
    dmat = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dmat, np.inf)
    return float(dmat.min(axis=1).mean())

def _sum_pairwise_metric(atoms: Atoms) -> float:
    """Sum of all unique pairwise distances under PBC in x,y."""
    dmat = atoms.get_all_distances(mic=True)
    i, j = np.triu_indices(len(atoms), k=1)
    return float(dmat[i, j].sum())

def _repulsive_energy_metric(atoms: Atoms, sigma: float=1.0, exponent: int=12) -> float:
    """
    Purely repulsive inverse-power energy:
      E = Σ_ij (σ / r_ij)^exponent
    Lower is better (fewer close contacts).
    """
    dmat = atoms.get_all_distances(mic=True)
    i, j = np.triu_indices(len(atoms), k=1)
    rij = dmat[i, j]
    rij = np.clip(rij, 1e-6, None)
    return float((sigma / rij)**exponent).sum()

def is_new_atoms_better(
        atoms1: Atoms,
        atoms2: Atoms,
        method: str = 'avg_nn',
        **kwargs
) -> bool:
    """
    Compare two ASE Atoms objects by a chosen packing metric.

    Args:
        atoms1, atoms2: The two structures to compare.
        method: One of
            - 'min'     : maximize minimum distance
            - 'avg_nn'  : maximize average nearest-neighbor distance
            - 'sum'     : maximize sum of all pairwise distances
            - 'energy'  : minimize repulsive inverse-power energy
        **kwargs: extra parameters for metric (e.g. sigma, exponent for 'energy').

    Returns:
        True if atoms2 is “better” than atoms1 under the chosen metric.
    """
    methods = {
        'min':    _min_dist_metric,
        'avg_nn': _avg_nn_metric,
        'sum':    _sum_pairwise_metric,
        'energy': lambda at: _repulsive_energy_metric(at, **kwargs),
    }
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; choose from {list(methods)}")

    score1 = methods[method](atoms1)
    score2 = methods[method](atoms2)

    # for 'energy', lower is better; for others, higher is better
    if method == 'energy':
        return score2 < score1
    else:
        return score2 > score1

class PlaneGroup:
    """
    Represents a 2D wallpaper (plane) group, allowing cell and atom generation.

    Attributes:
        pg_number (int): The group number 1–17.
        pg_symbol (str): The standard Hermann–Mauguin symbol.
        wyckoff_letters (List[str]): Available Wyckoff letters for this group.
    """

    _SYMBOLS = [
        'p1', 'p2', 'pm', 'pg', 'cm',
        'pmm', 'pmg', 'pgg', 'cmm',
        'p4', 'p4m', 'p4g',
        'p3', 'p3m1', 'p31m',
        'p6', 'p6m',
    ]
    _NUMBER_TO_SYMBOL = {i+1: s for i, s in enumerate(_SYMBOLS)}
    _SYMBOL_TO_NUMBER = {s: i+1 for i, s in enumerate(_SYMBOLS)}

    def __init__(self, plane_group: Union[int, str]) -> None:
        """
        Create a PlaneGroup by number or symbol.

        Args:
            plane_group: Integer 1–17 or a string symbol like 'p4m'.

        Raises:
            ValueError: If the input is not a valid group number or symbol.
        """
        if isinstance(plane_group, int):
            if plane_group not in self._NUMBER_TO_SYMBOL:
                raise ValueError(f"Invalid plane group number: {plane_group}")
            self.pg_number = plane_group
            self.pg_symbol = self._NUMBER_TO_SYMBOL[plane_group]

        elif isinstance(plane_group, str):
            sym = plane_group.lower()
            if sym not in self._SYMBOL_TO_NUMBER:
                raise ValueError(f"Invalid plane group symbol: '{plane_group}'")
            self.pg_number = self._SYMBOL_TO_NUMBER[sym]
            self.pg_symbol = sym

        else:
            raise TypeError("plane_group must be int (1–17) or str like 'p4m'")

        self.wyckoff_letters: List[str] = list(wyckoff_pos[self.pg_number].keys())

    def generate_uc(self) -> Atoms:
        """
        Generate an ASE Atoms instance for the unit cell, placing carbon
        atoms at the last Wyckoff position.

        Returns:
            An ASE Atoms object with symbols 'C' and positions in the cell.
        """
        cell = generate_plane_group_cell(self.pg_number)
        letter = self.wyckoff_letters[-1]
        wp = WyckoffPosition(self.pg_number, letter)
        coords = wp.generate_positions(return_z=True)
        symbols = ["C"] * len(coords)
        return Atoms(symbols=symbols, scaled_positions=coords, cell=cell, pbc=[True, True, False])

    def generate_unit_cell(self, structure_dict, cell=None, thickness=24.):
        if cell is None:
            cell = generate_plane_group_cell(self.pg_number, c=thickness)

        atom_symbols = []
        scaled_pos = []
        for symbol, letters in structure_dict.items():
            for letter in letters:
                wp = WyckoffPosition(self.pg_number, letter)
                xyz = wp.generate_positions(return_z=True)
                scaled_pos.append(xyz)
                atom_symbols.append([symbol] * len(xyz))
        atom_symbols = np.hstack(atom_symbols)
        scaled_pos = np.vstack(scaled_pos)
        atoms = Atoms(symbols=atom_symbols,
                      scaled_positions=scaled_pos,
                      cell=cell, pbc=[True, True, False])
        return atoms

    def generate_unit_cell_with_sampling(self, structure_dict, thickness=12., samples=10):
        cell = generate_plane_group_cell(self.pg_number, c=thickness)

        atoms = self.generate_unit_cell(structure_dict, cell=cell, thickness=thickness)
        for i in range(samples):
            atoms_ = self.generate_unit_cell(structure_dict, cell=cell, thickness=thickness)
            if is_new_atoms_better(atoms, atoms_):
                atoms = atoms_.copy()
        return atoms

