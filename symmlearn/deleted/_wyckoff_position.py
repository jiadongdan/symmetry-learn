import numpy as np
from ase.cell import Cell
from ase import Atoms
from typing import (
    List, Tuple,
    Optional, Union
)

# Wyckoff definitions for 2D plane groups
wyckoff_pos = {
    1: {'a': ['x, y']},
    2: {'a': ['0, 0'],
        'b': ['0, 1/2'],
        'c': ['1/2, 0'],
        'd': ['1/2, 1/2'],
        'e': ['x, y', '-x, -y']},
    3: {'a': ['0, y'],
        'b': ['1/2, y'],
        'c': ['x, y', '-x, y']},
    4: {'a': ['x, y', '-x, y+1/2']},
    5: {'a': ['0, y', '1/2, y+1/2'],
        'b': ['x, y', 'x, -y', 'x+1/2, y+1/2', '-x+1/2, y+1/2']},
    6: {'a': ['0, 0'],
        'b': ['0, 1/2'],
        'c': ['1/2, 0'],
        'd': ['1/2, 1/2'],
        'e': ['x, 0', '-x, 0'],
        'f': ['x, 1/2', '-x,  1/2'],
        'g': ['0, y', '0, -y'],
        'h': ['1/2, y', '1/2, -y'],
        'i': ['x, y', '-x, -y', '-x, y', 'x, -y']},
    7: {'a': ['0, 0', '1/2, 0'],
        'b': ['0, 1/2', '1/2, 1/2'],
        'c': ['1/4, y', '3/4, -y'],
        'd': ['x, y', '-x, -y', '-x+1/2, y', 'x+1/2, -y']},
    8: {'a': ['0, 0', '1/2, 1/2'],
        'b': ['1/2, 0', '0, 1/2'],
        'c': ['x, y', '-x, -y', '-x+1/2, y+1/2', 'x+1/2, -y+1/2']},
    9: {'a': ['0, 0', '1/2, 1/2'],
        'b': ['0, 1/2', '1/2, 1'],
        'c': ['1/4, 1/4', '3/4, 1/4', '3/4, 3/4', '5/4, 3/4'],
        'd': ['x, 0', '-x, 0', 'x+1/2, 1/2', '-x+1/2, 0'],
        'e': ['0, y', '0, -y', '1/2, y+1/2', '1/2, -y+1/2'],
        'f': ['x, y', '-x, -y', '-x, y', 'x, -y', 'x+1/2, y+1/2', '-x+1/2, -y+1/2', '-x+1/2, y+1/2', 'x+1/2, -y+1/2']},
    10:{'a': ['0, 0'],
        'b': ['1/2, 1/2'],
        'c': ['1/2, 0', '0, 1/2'],
        'd': ['x, y', '-x, -y', '-y, x', 'y, -x']},
    11:{'a': ['0, 0'],
        'b': ['1/2, 1/2'],
        'c': ['1/2, 0', '0, 1/2'],
        'd': ['x, 0', '-x, 0', '0, x', '0, -x'],
        'e': ['x, 1/2', '-x, 1/2', '1/2, x', '1/2, -x'],
        'f': ['x, x', '-x, -x', '-x, x', 'x, -x'],
        'g': ['x, y', '-x, -y', '-y, x', 'y, -x', '-x, y', 'x, -y', 'y, x', '-y, -x']},
    12:{'a': ['0, 0', '1/2, 1/2'],
        'b': ['1/2, 0', '0, 1/2'],
        'c': ['x, x+1/2', '-x, x+1/2', '-x+1/2, x', 'x+1/2, -x'],
        'd': ['x, y', '-x, -y', '-y, x', 'y, -x', '-x+1/2, y+1/2', 'x+1/2, -y+1/2', 'y+1/2, x+1/2', '-y+1/2, -x+1/2']},
    13:{'a': ['0, 0'],
        'b': ['1/3, 2/3'],
        'c': ['2/3, 1/3'],
        'd': ['x, y', '-y, x-y', '-x+y, -x']},
    14:{'a': ['0, 0'],
        'b': ['1/3, 2/3'],
        'c': ['2/3, 1/3'],
        'd': ['x, -x', 'x, 2x', '-2x, -x'],
        'e': ['x, y', '-y, x-y', '-x+y, -x', '-y, x', '-x+y, y', 'x, x-y']},
    15:{'a': ['0, 0'],
        'b': ['1/3, 2/3', '2/3, 1/3'],
        'c': ['x, 0', '0, x', '-x, -x'],
        'd': ['x, y', '-y, x-y', '-x+y, -x', 'y, x', 'x-y, -y', '-x, -x+y']},
    16:{'a': ['0, 0'],
        'b': ['1/3, 2/3', '2/3, 1/3'],
        'c': ['1/2, 0', '0, 1/2', '1/2, 1/2'],
        'd': ['x, y', '-y, x-y', '-x+y, -x', '-x, -y', 'y, -x+y', 'x-y, x']},
    17:{'a': ['0, 0'],
        'b': ['1/3, 2/3', '2/3, 1/3'],
        'c': ['1/2, 0', '0, 1/2', '1/2, 1/2'],
        'd': ['x, 0', '0, x', '-x, -x', '-x, 0', '0, -x', 'x, x'],
        'e': ['x, -x', 'x, 2x', '-2x, -x', '-x, x', '-x, -2x', '2x, x'],
        'f': ['x, y', '-y, x-y', '-x+y, -x', '-x, -y', 'y, -x+y', 'x-y, x', '-y, -x', '-x+y, y', 'x, x-y', 'y, x', 'x-y, -y', '-x, -x+y']}
}

class WyckoffPosition:
    def __init__(self,
                 plane_num: int,
                 letter: str,
                 wyckoff_dict: dict = wyckoff_pos):
        """
        Args:
            plane_num: 1–17 wallpaper group number.
            letter: Wyckoff letter in that group.
        Raises:
            ValueError: if group or letter invalid.
        """
        if plane_num not in wyckoff_dict:
            raise ValueError(f"Invalid plane group: {plane_num!r}")
        if letter not in wyckoff_dict[plane_num]:
            raise ValueError(f"Invalid Wyckoff letter: {letter!r}")

        self.plane_num = plane_num
        self.letter = letter
        raw_patterns: List[str] = wyckoff_dict[plane_num][letter]

        # Pre-compile each "x,y" → (code_x, code_y)
        self._compiled: List[Tuple[object, object]] = []
        for pat in raw_patterns:
            xs, ys = pat.split(",")
            # compile in 'eval' mode; will be evaluated later with {'x':..., 'y':...}
            code_x = compile(xs.strip(), "<wyckoff>", "eval")
            code_y = compile(ys.strip(), "<wyckoff>", "eval")
            self._compiled.append((code_x, code_y))

    def generate_positions(
            self,
            x: Optional[float] = None,
            y: Optional[float] = None,
            return_z = False,
            seed: Optional[Union[int, np.random.Generator]] = None
    ) -> np.ndarray:
        """
        Returns:
            (N,2) array of fractional coords, each wrapped into [0,1).
        """
        # 1) Set up RNG
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        # 2) Sample x, y if not provided
        x_val = x if x is not None else float(rng.random())
        y_val = y if y is not None else float(rng.random())

        # 3) Evaluate all compiled expressions
        out = []
        # minimal eval namespace
        eval_ns = {"__builtins__": None}
        local_vars = {"x": x_val, "y": y_val}

        for code_x, code_y in self._compiled:
            xv = eval(code_x, eval_ns, local_vars)
            yv = eval(code_y, eval_ns, local_vars)
            # wrap into [0,1)
            if return_z:
                out.append((xv % 1.0, yv % 1.0, 0.5))
            else:
                out.append((xv % 1.0, yv % 1.0))

        return np.asarray(out, dtype=float)


def generate_plane_group_cell(
        pg_number: int,
        a: float | None = None,
        b: float | None = None,
        gamma: float | None = None,
        a_range: tuple[float, float] = (0.5, 2.0),
        b_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = None
) -> Cell:
    """
    Return an ASE Cell for a 2D wallpaper (plane) group cell, with controlled
    sampling ranges for a and b.

    Parameters
    ----------
    pg_number : int
        Wallpaper group number (1–17).
    a : float or None
        x‐lattice constant; if None, sampled from U(a_range[0], a_range[1]).
    b : float or None
        y‐lattice constant; if None:
          - for oblique/rectangular: sampled from U(b_range[0], b_range[1])
          - otherwise set equal to a.
    gamma : float or None
        Angle between a and b (deg); if None:
          - pg 1: sampled from U(60,120)
          - rectangular/square: 90
          - trigonal: 120
          - hexagonal: 60
    a_range : (float, float)
        Min and max for sampling a when a is None.
    b_range : (float, float)
        Min and max for sampling b when b is None.
    seed : int or None
        RNG seed for reproducible sampling of a, b, and γ.

    Returns
    -------
    ase.cell.Cell
        Cell whose rows are the three lattice vectors.
    """
    rng = np.random.default_rng(seed)

    # sample or assign a
    if a is None:
        a = float(rng.uniform(a_range[0], a_range[1]))

    # Bravais‐type sets
    rectangular = set(range(2, 10))
    square      = {10, 11}
    trigonal    = {12, 13, 14}
    hexagonal   = {15, 16, 17}

    # sample or assign b
    if b is None:
        if pg_number == 1 or pg_number in rectangular:
            b = float(rng.uniform(b_range[0], b_range[1]))
        else:
            b = a

    # sample or assign gamma
    if gamma is None:
        if pg_number == 1:
            gamma = float(rng.uniform(60.0, 120.0))
        elif pg_number in rectangular or pg_number in square:
            gamma = 90.0
        elif pg_number in trigonal:
            gamma = 120.0
        elif pg_number in hexagonal:
            gamma = 60.0
        else:
            raise ValueError(f"Plane group must be 1–17; got {pg_number}")

    # convert to radians
    gamma_rad = np.deg2rad(gamma)

    # build the 3×3 lattice matrix
    lattice = np.array([
        [a,                    0.0,                    0.0],
        [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0.0],
        [0.0,                  0.0,                    1.0],  # unit z-vector
    ], dtype=float)

    return Cell(lattice)


class PlaneGroup:
    """
    Represents one of the 17 wallpaper (plane) groups.
    Accepts either an int (1–17) or a string like 'p4m'.

    After initialization:
      - self.pg_number is an int in [1..17]
      - self.pg_symbol is the corresponding Hermann–Mauguin string
      - self.wyckoff_letters is a list of the Wyckoff letters for that group
    """

    # list of the 17 symbols, in numeric order
    _SYMBOLS = [
        'p1', 'p2', 'pm', 'pg', 'cm',
        'pmm', 'pmg', 'pgg', 'cmm',
        'p4', 'p4m', 'p4g',
        'p3', 'p3m1', 'p31m',
        'p6', 'p6m',
    ]
    _NUMBER_TO_SYMBOL = {i+1: s for i, s in enumerate(_SYMBOLS)}
    _SYMBOL_TO_NUMBER = {s: i+1 for i, s in enumerate(_SYMBOLS)}

    # Wyckoff letters for each plane group (from Bilbao Crystallographic Server)
    _WYCKOFF_LETTERS = {
        1:  ['a'],
        2:  ['a', 'b', 'c', 'd', 'e'],
        3:  ['a', 'b', 'c'],
        4:  ['a'],
        5:  ['a', 'b'],
        6:  ['a', 'b', 'c', 'd', 'e'],
        7:  ['a', 'b', 'c', 'd', 'e'],
        8:  ['a', 'b', 'c'],
        9:  ['a', 'b', 'c', 'd', 'e'],
        10: ['a', 'b', 'c', 'd'],
        11: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        12: ['a', 'b', 'c', 'd', 'e', 'f'],
        13: ['a', 'b', 'c', 'd'],
        14: ['a', 'b', 'c', 'd', 'e', 'f'],
        15: ['a', 'b', 'c', 'd', 'e', 'f'],
        16: ['a', 'b', 'c'],
        17: ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
    }

    def __init__(self, plane_group):
        # resolve number/symbol
        if isinstance(plane_group, int):
            if plane_group not in self._NUMBER_TO_SYMBOL:
                raise ValueError(f"Invalid plane group number: {plane_group}")
            self.pg_number = plane_group
            self.pg_symbol = self._NUMBER_TO_SYMBOL[plane_group]

        elif isinstance(plane_group, str):
            key = plane_group.lower()
            if key not in self._SYMBOL_TO_NUMBER:
                raise ValueError(f"Invalid plane group symbol: '{plane_group}'")
            self.pg_number = self._SYMBOL_TO_NUMBER[key]
            self.pg_symbol = key

        else:
            raise TypeError("plane_group must be an int (1–17) or a string like 'p4m'")

        # now set the Wyckoff letters
        self.wyckoff_letters = list(wyckoff_pos[self.pg_number].keys())


    def generate_uc(self):
        cell = generate_plane_group_cell(self.pg_number)
        letter = self.wyckoff_letters[-1]
        wp = WyckoffPosition(self.pg_number, letter)
        xyz = wp.generate_positions(return_z=True)
        num_atoms = len(xyz)
        atoms = Atoms(
            symbols=["C"] * num_atoms,
            scaled_positions=xyz,
            cell=cell,
            pbc=[True, True, False],
        )
        return atoms
