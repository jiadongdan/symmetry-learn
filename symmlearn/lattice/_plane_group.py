from typing import List, Tuple, Optional, Union
import numpy as np
from ase.cell import Cell
from ase import Atoms
import spglib
from ..utils import show_atoms, check_random_state

from ._wyckoff_position import WyckoffPosition, wyckoff_pos
from ._utils import make_cell_square, rotate_atoms_xy_center, crop_atoms_xy_center
from ._pg_image import PGLattice
from ._mixin_plane_group import MixinShowPG, generate_plane_group_cell
from ._reduce_unit_cell_atoms import reduce_unit_cell_atoms, is_cell_same_size


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
    return ((sigma / rij)**exponent).sum()

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

def random_supercell(unit_cell, size):
    a = unit_cell.cellpar()[0]
    b = unit_cell.cellpar()[1]
    l = min(a, b)
    s = int(np.ceil(size / l) * 3)
    return (s, s, 1)

class PlaneGroup(MixinShowPG):
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

    

    def generate_unit_cell(
            self,
            structure_dict,
            cell: Optional[Cell] = None,
            a_range = (19, 30),
            thickness: float = 24.,
            seed: Optional[int] = None
    ) -> Atoms:
        """
        Generate an ASE Atoms instance for the unit cell with reproducible Wyckoff sampling.

        Args:
            structure_dict: Mapping of element symbols to Wyckoff letter lists.
            cell: An ASE Cell; if None, generated with given seed.
            thickness: z-axis cell length.
            seed: RNG seed for reproducibility.

        Returns:
            An ASE Atoms object with symbols and scaled positions.
        """
        rng = check_random_state(seed)
        if cell is None:
            cell = generate_plane_group_cell(self.pg_number, a_range=a_range, c=thickness, seed=seed)

        atom_symbols: List[str] = []
        scaled_pos_list: List[np.ndarray] = []

        for symbol, letters in structure_dict.items():
            for letter in letters:
                wp = WyckoffPosition(self.pg_number, letter)
                # derive a sub-seed for each Wyckoff call
                sub_seed = int(rng.integers(0, 2**32))
                coords = wp.generate_positions(return_z=True, seed=sub_seed)
                scaled_pos_list.append(coords)
                atom_symbols.extend([symbol] * len(coords))

        scaled_positions = np.vstack(scaled_pos_list)
        return Atoms(
            symbols=atom_symbols,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=[True, True, False]
        )

    def generate_unit_cell_with_sampling(
            self,
            structure_dict,
            cell = None,
            a_range = (19, 30),
            thickness: float = 12.,
            max_samples: int = 10,
            metric_method: str = 'avg_nn',
            seed: Optional[int] = None
    ) -> Atoms:
        """
        Generate multiple unit cell samples and select the best by packing metric, with reproducible randomness.

        Args:
            structure_dict: Mapping of elements to Wyckoff letters.
            thickness: z-axis cell length.
            samples: Number of random samples.
            seed: RNG seed for reproducibility.

        Returns:
            The ASE Atoms object for the best-packed sample.
        """
        rng = check_random_state(seed)
        num_samples = rng.integers(1, max_samples, endpoint=True)
        # pre-generate unique seeds for each sample
        sample_seeds = rng.integers(0, 2**32, size=num_samples)

        # first sample
        best_atoms = self.generate_unit_cell(
            structure_dict,
            cell=cell,
            a_range=a_range,
            thickness=thickness,
            seed=int(sample_seeds[0])
        )

        for ss in sample_seeds:
            atoms_candidate = self.generate_unit_cell(
                structure_dict,
                cell=best_atoms.cell,
                thickness=thickness,
                seed=int(ss)
            )
            if is_new_atoms_better(best_atoms, atoms_candidate, method=metric_method):
                best_atoms = atoms_candidate.copy()

        return best_atoms

    def generate_lattice(self,
                         structure_dict,
                         cell = None,
                         a_range = (19, 30),
                         size = 512,
                         thickness: float = 12.,
                         max_samples: int = 10,
                         angle_deg = None,
                         sigma_method: str = 'mean',
                         metric_method: str = 'avg_nn',
                         seed: Optional[int] = None,
                         debug = False,
        ) -> Atoms:
        rng = check_random_state(seed)
        atoms_unit_cell_ = self.generate_unit_cell_with_sampling(structure_dict=structure_dict,
                                                                cell=cell,
                                                                a_range=a_range,
                                                                thickness=thickness,
                                                                max_samples=max_samples,
                                                                metric_method=metric_method,
                                                                seed=rng)
        atoms_unit_cell, pg_num_new = reduce_unit_cell_atoms(atoms_unit_cell_)

        if pg_num_new != self.pg_number:
            print('Plane Group number has been updated from {} to {}'.format(self.pg_number, pg_num_new))
        if not is_cell_same_size(atoms_unit_cell_.cell, atoms_unit_cell.cell):
            print('Unit cell has been updated.')

        supercell = random_supercell(atoms_unit_cell.get_cell(), size)
        # print(supercell)
        if angle_deg is None:
            angle_deg = rng.uniform(0, 360)

        atoms_unit_cell.rotate('z', angle_deg, rotate_cell=True) # rotate unit cell atoms
        atoms = atoms_unit_cell * supercell                      # atoms grow
        if debug:
            show_atoms(atoms)
        atoms = make_cell_square(atoms)                          # make atoms square
        if debug:
            show_atoms(atoms)
        atoms = crop_atoms_xy_center(atoms, a_new=size, b_new=size)
        if debug:
            show_atoms(atoms)

        return PGLattice(pg_number=pg_num_new,
                         atoms=atoms,
                         unit_cell_atoms=atoms_unit_cell,
                         sigma_method=sigma_method)
