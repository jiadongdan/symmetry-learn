import numpy as np
import random
from fractions import Fraction

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


class WyckoffPosition1:

    def __init__(self, plane_num: int, letter: str):
        """
        Args:
            plane_num: int from 1–17 indicating the wallpaper group.
            letter: Wyckoff letter (e.g., 'a', 'b', ...) in that group.

        Raises:
            ValueError: for invalid group number or letter.
        """
        # Validate inputs
        if plane_num not in wyckoff_pos:
            raise ValueError(f"Invalid plane group: {plane_num!r}")
        valid_letters = wyckoff_pos[plane_num].keys()
        if letter not in valid_letters:
            raise ValueError(f"Invalid Wyckoff letter: {letter!r}; choose from {list(valid_letters)}")

        self.plane_num = plane_num
        self.letter = letter
        # coords: a list of string
        coords = wyckoff_pos[plane_num][letter]
        self.coords= coords

    def generate_positions(self, x=None, y=None, seed=None):
        """
        Generate the list of (x, y) positions for this Wyckoff letter.

        Args:
            x (float, optional): fixed 'x' value; if None, sampled randomly in [0,1).
            y (float, optional): fixed 'y' value; if None, sampled randomly in [0,1).
            seed (int or RNG-like, optional): seed or RNG object for reproducibility.

        Returns:
            List[Tuple[float, float]]: fractional positions modulo 1.
        """
        # 1) choose RNG
        if isinstance(seed, int):
            rng = random.Random(seed)
        elif seed is None:
            rng = random
        elif hasattr(seed, "random"):
            rng = seed
        else:
            raise ValueError("`seed` must be None, int, or an RNG-like object")

        # 2) determine x_val, y_val
        x_val = x if x is not None else rng.random()
        y_val = y if y is not None else rng.random()

        # 3) for each pattern string, eval the two expressions
        positions = []
        env = {"__builtins__": None}
        vars_ = {"x": x_val, "y": y_val}
        for pattern in self.coords:
            x_str, y_str = pattern.split(",")
            x_str, y_str = x_str.strip(), y_str.strip()
            # safe-ish eval in minimal namespace
            xv = eval(x_str, env, vars_)
            yv = eval(y_str, env, vars_)
            # wrap into [0,1)
            positions.append((xv % 1.0, yv % 1.0))

        return np.array(positions, dtype=float)
