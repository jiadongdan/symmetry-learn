import numpy as np
from itertools import combinations

from ._wyckoff_position import wyckoff_pos

def crystal_system_to_pg_num(system: str):
    """
    Map a 2D crystal system name to the list of plane-group (wallpaper-group) numbers.

    Parameters
    ----------
    system : str
        One of:
        "oblique", "rectangular", "square", "hexagonal".
        (Aliases accepted: monoclinic→oblique, orthorhombic→rectangular,
         tetragonal→square, trigonal→hexagonal.)

    Returns
    -------
    list of int
        Plane-group numbers associated with that crystal system.
    """
    system = system.strip().lower()

    mapping = {
        # oblique (parallelogram)
        "oblique": [1, 2],
        "parallelogram": [1, 2],
        "monoclinic": [1, 2],

        # rectangular + centered rectangular
        "rectangular": [3, 4, 5, 6, 7, 8, 9],
        "centered rectangular": [3, 4, 5, 6, 7, 8, 9],
        "orthorhombic": [3, 4, 5, 6, 7, 8, 9],

        # square
        "square": [10, 11, 12],
        "tetragonal": [10, 11, 12],

        # hexagonal / trigonal
        "hexagonal": [13, 14, 15, 16, 17],
        "trigonal": [13, 14, 15, 16, 17],
    }

    if system not in mapping:
        raise ValueError(f"Unknown crystal system '{system}'")

    return mapping[system]


def power_set(seq: list, exclude_empty: bool = True) -> list:
    """
    Return all subsets of `seq` (the power set).

    Args:
        seq: a list of items
        exclude_empty: if True, don’t include the empty list []
    Returns:
        a list of subsets (each subset is itself a list)
    """
    subsets = []
    start = 1 if exclude_empty else 0
    n = len(seq)
    for r in range(start, n + 1):
        for combo in combinations(seq, r):
            subsets.append(list(combo))
    return subsets


def random_structure_A(crystal_system, max_counts=12, seed=48):
    rng = np.random.default_rng(seed)
    pg_numbers = crystal_system_to_pg_num(crystal_system)
    pg_num = rng.choice(pg_numbers)

    letters = list(wyckoff_pos[pg_num].keys())
    ps = power_set(letters)
    counts = np.array([sum([len(wyckoff_pos[pg_num][k]) for k in ks]) for ks in ps])
    inds = np.where(counts <=max_counts)[0]
    ps = [ps[ind] for ind in inds]

    idx = rng.choice(range(len(ps)))

    return {'C': ps[idx]}, pg_num


