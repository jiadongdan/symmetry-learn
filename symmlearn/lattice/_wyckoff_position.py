import re
import numpy as np
from typing import List, Tuple, Optional, Union

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
        'b': ['x, y', '-x, y', 'x+1/2, y+1/2', '-x+1/2, y+1/2']},
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
        'c': ['1/4, 1/4', '3/4, 1/4', '3/4, 3/4', '1/4, 3/4'],
        'd': ['x, 0', '-x, 0', 'x+1/2, 1/2', '-x+1/2, 1/2'],
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
        'c': ['x, x+1/2', '-x, -x+1/2', '-x+1/2, x', 'x+1/2, -x'],
        'd': ['x, y', '-x, -y', '-y, x', 'y, -x', '-x+1/2, y+1/2', 'x+1/2, -y+1/2', 'y+1/2, x+1/2', '-y+1/2, -x+1/2']},
    13:{'a': ['0, 0'],
        'b': ['1/3, 2/3'],
        'c': ['2/3, 1/3'],
        'd': ['x, y', '-y, x-y', '-x+y, -x']},
    14:{'a': ['0, 0'],
        'b': ['1/3, 2/3'],
        'c': ['2/3, 1/3'],
        'd': ['x, -x', 'x, 2x', '-2x, -x'],
        'e': ['x, y', '-y, x-y', '-x+y, -x', '-y, -x', '-x+y, y', 'x, x-y']},
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
    """
    Handle Wyckoff letter definitions for a given 2D wallpaper group.
    """

    # Cache: plane-group → max multiplicity (general position multiplicity)
    _group_max_multiplicity = {}

    def __init__(
            self,
            pg_number: int,
            letter: str,
            wyckoff_dict: dict = wyckoff_pos
    ) -> None:

        if pg_number not in wyckoff_dict:
            raise ValueError(f"Invalid plane group: {pg_number!r}")
        if letter not in wyckoff_dict[pg_number]:
            raise ValueError(f"Invalid Wyckoff letter: {letter!r}")

        self.pg_number: int = pg_number
        self.letter: str = letter

        raw_patterns: List[str] = wyckoff_dict[pg_number][letter]
        self._patterns: List[str] = raw_patterns

        # Precompute multiplicity and whether this letter has free variables
        self._multiplicity: int = len(raw_patterns)
        # You can drop regex here; simple substring is enough for this dataset
        self._has_variables: bool = any(('x' in p) or ('y' in p) for p in raw_patterns)

        # Pre-compile each pattern "x_expr, y_expr"
        self._compiled: List[Tuple[object, object]] = []
        for pat in raw_patterns:
            pat = re.sub(r'(-?\d+)([xy])', r'\1*\2', pat)
            xs, ys = pat.split(",")
            code_x = compile(xs.strip(), "<wyckoff>", "eval")
            code_y = compile(ys.strip(), "<wyckoff>", "eval")
            self._compiled.append((code_x, code_y))

    # ---------- helpers ----------

    @property
    def multiplicity(self) -> int:
        return self._multiplicity

    def _group_max_mult(self) -> int:
        """Return cached maximum multiplicity for this plane group."""
        pg = self.pg_number
        if pg not in self._group_max_multiplicity:
            group_dict = wyckoff_pos[pg]
            self._group_max_multiplicity[pg] = max(len(pats) for pats in group_dict.values())
        return self._group_max_multiplicity[pg]

    # ---------- classification API ----------

    def is_general(self) -> bool:
        """
        True if this Wyckoff position is the general position
        (has maximum multiplicity in its plane group).
        """
        return self.multiplicity == self._group_max_mult()

    def is_special_fixed(self) -> bool:
        """
        True if this Wyckoff position is special with no free parameters
        (no 'x' or 'y' anywhere in its coordinate expressions).
        """
        # No free variables → fixed
        return not self._has_variables

    def is_special_variable(self) -> bool:
        """
        True if this Wyckoff position is special with variable coordinates:
        it has free parameters, but is not the general position.
        """
        return self._has_variables and not self.is_general()

    def classify(self) -> str:
        if self.is_general():
            return "general"
        if self.is_special_fixed():
            return "special_fixed"
        if self.is_special_variable():
            return "special_variable"
        raise RuntimeError("Unreachable classification state.")


    def generate_positions(
            self,
            x: Optional[float] = None,
            y: Optional[float] = None,
            return_z: bool = False,
            seed: Optional[Union[int, np.random.Generator]] = None
    ) -> np.ndarray:
        """
        Generate fractional coordinates for all positions of this Wyckoff letter.

        Args:
            x: Fractional x parameter; if None, randomly sampled in [0,1).
            y: Fractional y parameter; if None, randomly sampled in [0,1).
            return_z: If True, include a z coordinate of 0.5 for each position.
            seed: RNG seed or numpy Generator for reproducible sampling.

        Returns:
            An (N,2) array of fractional (x,y) positions (or (N,3) if return_z=True).
        """
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed)

        x_val = x if x is not None else float(rng.random())
        y_val = y if y is not None else float(rng.random())

        eval_ns = {"__builtins__": None}
        local_vars = {"x": x_val, "y": y_val}

        out = []
        for code_x, code_y in self._compiled:
            xv = eval(code_x, eval_ns, local_vars)
            yv = eval(code_y, eval_ns, local_vars)
            if return_z:
                out.append((xv % 1.0, yv % 1.0, 0.5))
            else:
                out.append((xv % 1.0, yv % 1.0))

        return np.asarray(out, dtype=float)

class WyckoffStructure:

    def __init__(self, pg_num, structure_letters):
        self.structure_letters = structure_letters
        self.pg_num = pg_num
        self.unique_atoms = len(structure_letters)

    @property
    def num_atoms(self):
        return sum([len(wyckoff_pos[self.pg_num][letter]) for letters in self.structure_letters for letter in letters])
