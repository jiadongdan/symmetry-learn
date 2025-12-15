from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ase.cell import Cell


def generate_plane_group_cell(
        pg_number: int,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = 12,
        gamma: Optional[float] = None,
        a_range: Tuple[float, float] = (2.0, 4.0),
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
    # 2D Bravais lattice classes
    oblique = {1, 2}
    rectangular = set(range(3, 10))
    square = {10, 11, 12}
    hexagonal = set(range(13, 18))

    rng = np.random.default_rng(seed)
    # Sample a if needed
    if a is None:
        a = float(rng.uniform(a_range[0], a_range[1]))

    # Sample or set b
    if b is None:
        if pg_number in oblique or pg_number in rectangular:
            b = float(rng.uniform(0.5 * a, 0.9 * a))
        else:
            b = a

    # Sample or set gamma
    if gamma is None:
        if pg_number in oblique:
            gamma = float(rng.uniform(60.0, 120.0))
        elif pg_number in rectangular or pg_number in square:
            gamma = 90.0
        elif pg_number in hexagonal:
            gamma = 120.0   # To match ITC/Bilbao tables
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


def transform_via_cell(data, cell):
    """
    Transform 2D fractional coordinates (f_a, f_b) to Cartesian coordinates.

    Supports arbitrary leading dimensions, e.g. (N,2), (N,M,2), (N,M,K,2).

    Parameters
    ----------
    data : array_like, shape (..., 2)
        Fractional coordinates.
    cell : ase.cell.Cell or array_like
        Simulation cell: 3x3 or 2x2.

    Returns
    -------
    xy : np.ndarray, shape (..., 2)
        Cartesian coordinates.
    """
    if data is None:
        return None

    frac = np.asarray(data, dtype=float)
    cell_arr = np.asarray(cell, dtype=float)

    if frac.shape[-1] != 2:
        raise ValueError(
            f"`data` must have fractional coords of length 2 in the last axis, "
            f"got last axis {frac.shape[-1]}."
        )

    # Extract 2D basis (a and b vectors)
    if cell_arr.shape == (3, 3):
        # Take a, b vectors and project to xy-plane
        basis2d = cell_arr[:2, :2]    # shape (2, 2)
    elif cell_arr.shape == (2, 2):
        basis2d = cell_arr
    else:
        raise ValueError(
            f"`cell` must be 3x3 or 2x2, got shape {cell_arr.shape}."
        )

    # Perform vectorized transformation:
    # (..., 2) @ (2, 2) → (..., 2)
    return frac @ basis2d


def add_rotational_centers(ax, pts, n_fold, **kwargs):
    if pts is None:
        return
    if n_fold == 2:
        ax.scatter(pts[:, 0], pts[:, 1], marker='d', **kwargs)
    elif n_fold == 3:
        ax.scatter(pts[:, 0], pts[:, 1], marker='^', **kwargs)
    elif n_fold == 4:
        ax.scatter(pts[:, 0], pts[:, 1], marker='D', **kwargs)
    elif n_fold == 6:
        ax.scatter(pts[:, 0], pts[:, 1], marker='h', **kwargs)


def add_lines(ax, pairs, **kwargs):
    """
    Add multiple line segments to an axis from pairs of points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw on.
    pairs : array_like, shape (..., 2, 2)
        Pairs of points. The last two dimensions must be (2, 2):
        [ [x0, y0], [x1, y1] ] per line segment.
        Examples:
          - (N, 2, 2) for N line segments
          - (M, N, 2, 2) etc. (will be flattened to a list of segments)
    **kwargs :
        Passed directly to LineCollection (e.g. color, linewidth, alpha).

    Returns
    -------
    lc : matplotlib.collections.LineCollection
        The created LineCollection instance.
    """
    if pairs is None:
        return

    segments = np.asarray(pairs, dtype=float)

    if segments.shape[-2:] != (2, 2):
        raise ValueError(
            f"`pairs` must have shape (..., 2, 2) for line endpoints, "
            f"got {segments.shape}"
        )

    # Flatten any leading dimensions to get a simple (N, 2, 2) array of segments
    segments = segments.reshape(-1, 2, 2)

    lc = LineCollection(segments, **kwargs)
    ax.add_collection(lc)

    # Optionally autoscale view to include all line segments
    ax.autoscale_view()

    return lc


Point = Tuple[float, float]
Pair  = Tuple[Point, Point]

@dataclass(frozen=True)
class PlaneGroupPattern:
    P2: Optional[Sequence[Point]] = None
    P3: Optional[Sequence[Point]] = None
    P4: Optional[Sequence[Point]] = None
    P6: Optional[Sequence[Point]] = None
    mirror_pairs: Optional[Sequence[Pair]] = None
    glide_pairs: Optional[Sequence[Pair]] = None

    # class-level constant shared by all patterns
    outline_pairs: Sequence[Pair] = (
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 1.0), (0.0, 1.0)),
    )


PG_PATTERNS: Dict[int, PlaneGroupPattern] = {
    10: PlaneGroupPattern(
        P2=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
        P4=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)],
        # others default to None
    ),
    11: PlaneGroupPattern(
        P2=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
        P4=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)],
        mirror_pairs=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 1.0), (1.0, 1.0)),
            ((1.0, 1.0), (1.0, 0.0)),
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 1.0)), # diagonal
            ((0.0, 1.0), (1.0, 0.0)), # diagonal
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (0.5, 1.0)),
        ],
        glide_pairs=[
            ((0.0, 0.5), (0.5, 1.0)),
            ((0.5, 1.0), (1.0, 0.5)),
            ((1.0, 0.5), (0.5, 0.0)),
            ((0.5, 0.0), (0.0, 0.5)),
        ],
    ),
    12: PlaneGroupPattern(
        P2=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
        P4=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)],
        mirror_pairs=[
            ((0.0, 0.5), (0.5, 1.0)),
            ((0.5, 1.0), (1.0, 0.5)),
            ((1.0, 0.5), (0.5, 0.0)),
            ((0.5, 0.0), (0.0, 0.5)),
        ],
        glide_pairs=[
            ((0.25, 0.0), (0.25, 1.0)),
            ((0.75, 0.0), (0.75, 1.0)),
            ((0.0, 0.25), (1.0, 0.25)),
            ((0.0, 0.75), (1.0, 0.75)),
            ((0.0, 0.0), (1.0, 1.0)), # diagonal
            ((0.0, 1.0), (1.0, 0.0)), # diagonal
        ],
    ),
    17: PlaneGroupPattern(
        P2=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5), (0.5, 0.5)],
        P3=[(1/3, 2/3), (2/3, 1/3)],
        P6=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        mirror_pairs=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 1.0)),
        ],
        glide_pairs=[
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (0.5, 1.0)),
        ],
    ),
}



class MixinShowPG:
    def show(self, ax=None, seed=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        # use the provided seed instead of hard-coding None
        cell = generate_plane_group_cell(self.pg_number, seed=seed)

        try:
            pattern = PG_PATTERNS[self.pg_number]
        except KeyError:
            raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

        # transform all coordinate sets through the cell
        P2 = transform_via_cell(pattern.P2, cell)
        P3 = transform_via_cell(pattern.P3, cell)
        P4 = transform_via_cell(pattern.P4, cell)
        P6 = transform_via_cell(pattern.P6, cell)
        mirror_pairs  = transform_via_cell(pattern.mirror_pairs, cell)
        glide_pairs   = transform_via_cell(pattern.glide_pairs, cell)
        outline_pairs = transform_via_cell(pattern.outline_pairs, cell)

        # rotational centers
        add_rotational_centers(ax, P2, n_fold=2, color='#2d3742', zorder=5, s=100)
        add_rotational_centers(ax, P3, n_fold=3, color='#2d3742', zorder=5, s=100)
        add_rotational_centers(ax, P4, n_fold=4, color='#2d3742', zorder=5, s=100)
        add_rotational_centers(ax, P6, n_fold=6, color='#2d3742', zorder=5, s=100)

        # mirrors & glides
        add_lines(ax, mirror_pairs, lw=2)
        add_lines(ax, glide_pairs, lw=1, ls='--')
        add_lines(ax, outline_pairs, lw=0.5, color='k')


        ax.axis('equal')
        ax.axis('off')

