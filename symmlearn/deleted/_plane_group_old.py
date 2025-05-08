import numpy as np
import random
from fractions import Fraction
from pyxtal.symmetry import Group

# Wyckoff definitions for 2D plane groups
wyckoff_pos = {
    1: {'a': 'x, y'},
    2: {'a': '0, 0', 'b': '0, 1/2', 'c': '1/2, 0', 'd': '1/2, 1/2', 'e': 'x, y'},
    3: {'a': '0, y', 'b': '1/2, y', 'c': 'x, y'},
    4: {'a': 'x, y'},
    5: {'a': '0, y', 'b': 'x, y'},
    6: {'a': '0, 0', 'b': '0, 1/2', 'c': '1/2, 0', 'd': '1/2, 1/2',
        'e': 'x, 0', 'f': 'x, 1/2', 'g': '0, y', 'h': '1/2, y', 'i': 'x, y'},
    7: {'a': '0, 0', 'b': '0, 1/2', 'c': '1/4, y', 'd': 'x, y'},
    8: {'a': '0, 0', 'b': '1/2, 0', 'c': 'x, y'},
    9: {'a': '0, 0', 'b': '0, 1/2', 'c': '1/4, 1/4', 'd': 'x, 0',
        'e': '0, y', 'f': 'x, y'},
    10:{'a': '0, 0', 'b': '1/2, 1/2', 'c': '1/2, 0', 'd': 'x, y'},
    11:{'a': '0, 0', 'b': '1/2, 1/2', 'c': '1/2, 0', 'd': 'x, 0', 'e': 'x, 1/2',
        'f': 'x, x', 'g': 'x, y'},
    12:{'a': '0, 0', 'b': '1/2, 0', 'c': 'x, x+1/2', 'd': 'x, y'},
    13:{'a': '0, 0', 'b': '1/3, 2/3', 'c': '2/3, 1/3', 'd': 'x, y'},
    14:{'a': '0, 0', 'b': '1/3, 2/3', 'c': '2/3, 1/3', 'd': 'x, -x', 'e': 'x, y'},
    15:{'a': '0, 0', 'b': '1/3, 2/3', 'c': 'x, 0', 'd': 'x, y'},
    16:{'a': '0, 0', 'b': '1/3, 2/3', 'c': '1/2, 0', 'd': 'x, y'},
    17:{'a': '0, 0', 'b': '1/3, 2/3', 'c': '1/2, 0', 'd': 'x, 0', 'e': 'x, -x',
        'f': 'x, -x', 'g': 'x, y'}
}

class WyckoffPosition1:

    # one-to-one map from the 17 wallpaper groups → layer groups in PyXtal
    PLANE_TO_LAYER = {
        1:  1,   2:  3,   3: 11,   4:  9,   5: 10,
        6: 23,   7: 24,   8: 25,   9: 26,  10: 49,
        11: 55,  12: 52,  13: 65,  14: 69,  15: 70,
        16: 73,  17: 77,
    }

    def __init__(self, plane_num: int, letter: str):
        if plane_num not in wyckoff_pos:
            raise ValueError(f"Invalid plane group: {plane_num!r}")
        self.plane_num = plane_num

        valid_letters = list(wyckoff_pos[self.plane_num].keys())
        if letter not in valid_letters:
            raise ValueError(f"Invalid Wyckoff letter: {letter!r}; choose from {valid_letters}")
        self.letter = letter

        coords_text = wyckoff_pos[self.plane_num][self.letter]
        self.coords_text = coords_text

        parts = [p.strip() for p in coords_text.split(',')]
        if len(parts) != 2:
            raise ValueError(f"Unexpected coords format: {coords_text!r}")
        x_str, y_str = parts

        # Number of free variables
        self.dof = int(x_str == 'x') + int(y_str == 'y')

        # Fixed coordinate values, if any
        fixed = {}
        if x_str not in ('x', 'y'):
            fixed['x'] = float(Fraction(x_str))
        if y_str not in ('x', 'y'):
            fixed['y'] = float(Fraction(y_str))
        self.fixed_coords = fixed

        self.free_axes = [ax for ax in ('x', 'y') if ax not in self.fixed_coords]

        self.layer_num = self.PLANE_TO_LAYER[plane_num]
        # initialize the layer group
        layer_group = Group(self.layer_num, dim=2)
        all_wps = {wp.get_label()[-1]:wp for wp in layer_group.Wyckoff_positions}
        self.wp = all_wps[self.letter]


    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}"
                f"(plane_num={self.plane_num}, letter={self.letter!r}, dof={self.dof})")

    def __str__(self) -> str:
        return (f"Wyckoff position {self.letter} "
                f"in plane group {self.plane_num} (dof={self.dof})")

    def generate_positions(self,
                           x: float = None,
                           y: float = None,
                           seed=None) -> tuple[float, float]:
        """
        Generate (x, y) coordinates for this Wyckoff position.

        Args:
            x (float, optional): override x-coordinate.
            y (float, optional): override y-coordinate.
            seed (None, int, or RNG):
                - None: use module-level random
                - int: seed a new random.Random(seed)
                - RNG: any object with .random(), e.g. random.Random or numpy Generator

        Returns:
            Tuple of (x_val, y_val).
        """
        # Resolve RNG from seed
        if isinstance(seed, int):
            rng = random.Random(seed)
        elif seed is None:
            rng = random
        elif hasattr(seed, 'random'):
            rng = seed
        else:
            raise ValueError("`seed` must be None, an int, or an object with a .random() method")

        # Apply fixed coords first
        if 'x' in self.fixed_coords:
            x_val = self.fixed_coords['x']
        elif x is not None:
            x_val = float(x)
        else:
            x_val = rng.random()

        if 'y' in self.fixed_coords:
            y_val = self.fixed_coords['y']
        elif y is not None:
            y_val = float(y)
        else:
            y_val = rng.random()

        # self.wp.get_all_positions only works for (x, y, z)
        xyz = (x_val, y_val, 0.)
        return self.wp.get_all_positions(xyz)[:, 0:2]


class WyckoffPosition:
    # one-to-one map from the 17 wallpaper groups → layer groups in PyXtal
    PLANE_TO_LAYER = {
        1:  1,   2:  3,   3: 11,   4: 12,   5: 13,
        6: 23,   7: 24,   8: 25,   9: 26,  10: 49,
        11: 55,  12: 52,  13: 65,  14: 69,  15: 70,
        16: 73,  17: 77,
    }

    def __init__(self, plane_num, letter):
        self.plane_num = plane_num
        layer_num = self.PLANE_TO_LAYER[plane_num]
        # initialize the 2D layer (wallpaper) group
        layer_group = Group(layer_num, dim=2)
        all_sites = {wp.get_label()[-1]:wp for wp in layer_group.Wyckoff_positions}
        self.valid_letter = [wp.get_label()[-1] for wp in layer_group.Wyckoff_positions]
        self.dof = [max(wp.get_dof() - 1, 0) for wp in layer_group.Wyckoff_positions]
        # letter has to be in self.valid_letters
        self.letter = letter
        if self.dof == 2:
            self.x = 'random'
            self.y = 'random'
        elif self.dof == 1:
            wp = all_sites[self.letter]
            if wp.get_frozen_axis()[0] == 0:
                self.x = 'some_value'
                self.y = 'random'
            elif wp.get_frozen_axis()[0] == 1:
                self.x = 'random'
                self.y = 'some_value'
        elif self.dof == 0:
            self.random_x = 'some_value'
            self.radom_y = 'some_value'


class PlaneGroup:
    # one-to-one map from the 17 wallpaper groups → layer groups in PyXtal
    PLANE_TO_LAYER = {
        1:  1,   2:  3,   3: 11,   4:  9,   5: 10,
        6: 23,   7: 24,   8: 25,   9: 26,  10: 49,
        11: 55,  12: 52,  13: 65,  14: 69,  15: 70,
        16: 73,  17: 77,
    }

    def __init__(self, plane_num):
        self.plane_num = plane_num
        layer_num = self.PLANE_TO_LAYER[plane_num]
        # initialize the 2D layer (wallpaper) group
        self.layer_group = Group(layer_num, dim=2)

        # build a list of site-descriptors
        self.sites = []
        for wp in self.layer_group.Wyckoff_positions:
            letter = wp.get_label()         # e.g. "1a", "2b", …
            raw_dof = wp.get_dof()          # includes z-direction parameter
            # for plane groups, subtract the z-parameter
            dof = max(raw_dof - 1, 0)       # number of free planar fractional parameters
            fixed = (dof == 0)              # no freedom → fixed site
            self.sites.append({
                'letter': letter,
                'dof':    dof,
                'fixed':  fixed,
                'wp':     wp,                # raw Wyckoff_position object
            })

    def summary(self):
        """Print a quick summary of all Wyckoff letters and their planar freedom."""
        for s in self.sites:
            status = "fixed" if s['fixed'] else f"{s['dof']} free param(s)"
            print(f"{s['letter']}: {status}")
