from dataclasses import dataclass
from typing import Tuple, Optional, Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ase.cell import Cell

from ._line_drawing import get_line
from ..utils import check_random_state


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

    rng = check_random_state(seed)
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
    mirror_pairs1: Optional[Sequence[Pair]] = None
    mirror_pairs2: Optional[Sequence[Pair]] = None
    glide_pairs: Optional[Sequence[Pair]] = None

    # class-level constant shared by all patterns
    outline_pairs: Sequence[Pair] = (
        ((0.0, 0.0), (0.0, 1.0)),
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 1.0), (1.0, 0.0)),
        ((1.0, 1.0), (0.0, 1.0)),
    )

    corner_pts: Optional[Sequence[Point]] = None
    hexagonal_edge_pts: Optional[Sequence[Point]] = None
    hexagonal_inside_pts: Optional[Sequence[Point]] = None
    square_edge_pts: Optional[Sequence[Point]] = None
    corner_center_pts: Optional[Sequence[Point]] = None
    corner_edge_center_pts: Optional[Sequence[Point]] = None
    rect_center_pts: Optional[Sequence[Point]] = None
    mirror_pairs_h: Optional[Sequence[Point]] = None
    mirror_pairs_v: Optional[Sequence[Point]] = None
    mirror_pairs_mid: Optional[Sequence[Point]] = None

PG_PATTERNS: Dict[int, PlaneGroupPattern] = {
    1: PlaneGroupPattern(),
    2: PlaneGroupPattern(
        P2 = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
              (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
              (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
    ),
    3: PlaneGroupPattern(
        mirror_pairs=[ # we should have vertical mirrors
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.5, 0.0), (0.5, 1.0)),
            ((1.0, 0.0), (1.0, 1.0)),
        ],
    ),
    4: PlaneGroupPattern(
        glide_pairs=[
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.0, 1.0), (1.0, 1.0)),
        ],
    ),
    5: PlaneGroupPattern(
        mirror_pairs=[   # we should have vertical mirrors
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.5, 0.0), (0.5, 1.0)),
            ((1.0, 0.0), (1.0, 1.0)),
        ],
        glide_pairs=[
            ((0.25, 0.0), (0.25, 1.0)),
            ((0.75, 0.0), (0.75, 1.0)),
        ],
    ),
    6: PlaneGroupPattern(
        P2 = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
              (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
              (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
        mirror_pairs=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 1.0), (1.0, 1.0)),
            ((1.0, 1.0), (1.0, 0.0)),
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (0.5, 1.0)),
        ],
    ),
    7: PlaneGroupPattern(
        P2 = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
              (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
              (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
        mirror_pairs=[ # we should have vertical mirrors
            ((0.25, 0.0), (0.25, 1.0)),
            ((0.75, 0.0), (0.75, 1.0)),
        ],
        glide_pairs=[ # we should have horizontal glides
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.0, 1.0), (1.0, 1.0)),
        ]
    ),
    8: PlaneGroupPattern(
        P2 = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
              (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
              (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
        glide_pairs=[
            ((0.0, 0.25), (1.0, 0.25)),
            ((0.0, 0.75), (1.0, 0.75)),
            ((0.25, 0.0), (0.25, 1.0)),
            ((0.75, 0.0), (0.75, 1.0)),
        ],
    ),
    9: PlaneGroupPattern(
        P2 = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
              (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
              (1.0, 0.0), (1.0, 0.5), (1.0, 1.0),
              (0.25, 0.25), (0.75, 0.25),(0.25, 0.75), (0.75, 0.75),
        ],
        mirror_pairs=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 1.0), (1.0, 1.0)),
            ((1.0, 1.0), (1.0, 0.0)),
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (0.5, 1.0)),
        ],
        glide_pairs=[
            ((0.0, 0.25), (1.0, 0.25)),
            ((0.0, 0.75), (1.0, 0.75)),
            ((0.25, 0.0), (0.25, 1.0)),
            ((0.75, 0.0), (0.75, 1.0)),
        ],
    ),
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
            ((0.0, 0.0), (1.0, 0.0)),
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
    13: PlaneGroupPattern(
        P3=[(1/3, 2/3), (2/3, 1/3), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    ),
    14: PlaneGroupPattern(
        P3=[(1/3, 2/3), (2/3, 1/3), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        mirror_pairs=[
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.5)),
            ((0.0, 0.0), (0.5, 1.0)),
            ((0.0, 0.5), (1.0, 1.0)),
            ((0.5, 0.0), (1.0, 1.0)),
        ],
        glide_pairs=[
            ((0.5, 0.0), (0.0, 0.5)),
            ((0.0, 0.5), (0.25, 1.0)),
            ((0.5, 0.0), (1, 0.25)),
            ((1.0, 0.5), (0.5, 1.0)),
            ((0.5, 1.0), (0.0, 0.75)),
            ((1.0, 0.5), (0.75, 0.0)),

            ((0.25, 0.0), (0.75, 1.0)),
            ((0.0, 0.25), (1.0, 0.75)),
        ],
    ),
    15: PlaneGroupPattern(
        P3=[(1/3, 2/3), (2/3, 1/3), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        mirror_pairs=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 0.0), (1.0, 1.0)),
            ((0.0, 1.0), (1.0, 1.0)),
        ],
        glide_pairs=[
            ((0.0,0.5), (1.0,0.5)),
            ((0.5,0.0), (0.5,1.0)),
            ((0.0, 0.5), (0.5, 1.0)),
            ((0.5, 0.0), (1.0, 0.5)),
        ],
    ),
    16: PlaneGroupPattern(
        P2=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5), (0.5, 0.5)],
        P3=[(1/3, 2/3), (2/3, 1/3)],
        P6=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
    ),
    17: PlaneGroupPattern(
        P2=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5), (0.5, 0.5)],
        P3=[(1/3, 2/3), (2/3, 1/3)],
        P6=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        mirror_pairs=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 0.0), (1.0, 1.0)),
            ((0.0, 1.0), (1.0, 1.0)),
            ((0.0, 1.0), (1.0, 0.0)),
            ((0.0, 0.0), (1.0, 0.5)),
            ((0.0, 0.0), (0.5, 1.0)),
            ((0.0, 0.5), (1.0, 1.0)),
            ((0.5, 0.0), (1.0, 1.0)),
        ],
        glide_pairs=[
            ((0.5, 0.0), (0.0, 0.5)),
            ((0.0, 0.5), (0.25, 1.0)),
            ((0.5, 0.0), (1, 0.25)),
            ((1.0, 0.5), (0.5, 1.0)),
            ((0.5, 1.0), (0.0, 0.75)),
            ((1.0, 0.5), (0.75, 0.0)),

            ((0.25, 0.0), (0.75, 1.0)),
            ((0.0, 0.25), (1.0, 0.75)),

            ((0.0,0.5), (1.0,0.5)),
            ((0.5,0.0), (0.5,1.0)),
            ((0.0, 0.5), (0.5, 1.0)),
            ((0.5, 0.0), (1.0, 0.5)),
        ],
    ),
    'hexagonal': PlaneGroupPattern(
        hexagonal_edge_pts=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5), (0.5, 0.5)],
        hexagonal_inside_pts=[(1/3, 2/3), (2/3, 1/3)],
        corner_pts=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        mirror_pairs1=[
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.5)),
            ((0.0, 0.0), (0.5, 1.0)),
            ((0.0, 0.5), (1.0, 1.0)),
            ((0.5, 0.0), (1.0, 1.0)),
        ],
        mirror_pairs2=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.0), (1.0, 1.0)),
            ((1.0, 0.0), (1.0, 1.0)),
            ((0.0, 1.0), (1.0, 1.0)),
        ],
    ),
    'square': PlaneGroupPattern(
        square_edge_pts=[(0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.0, 0.5)],
        corner_center_pts=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)],
        mirror_pairs1=[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 1.0), (1.0, 1.0)),
            ((1.0, 1.0), (1.0, 0.0)),
            ((1.0, 0.0), (0.0, 1.0)),
            ((0.0, 0.0), (1.0, 1.0)), # diagonal
            ((0.0, 1.0), (1.0, 0.0)), # diagonal
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.5, 0.0), (0.5, 1.0)),
        ],
        mirror_pairs2=[
            ((0.0, 0.5), (0.5, 1.0)),
            ((0.5, 1.0), (1.0, 0.5)),
            ((1.0, 0.5), (0.5, 0.0)),
            ((0.5, 0.0), (0.0, 0.5)),
        ],
    ),
    'rect': PlaneGroupPattern(
        corner_edge_center_pts = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
                                  (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
                                  (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
        rect_center_pts = [(0.25, 0.25), (0.75, 0.25),(0.25, 0.75), (0.75, 0.75)],
        mirror_pairs_h =[
            ((0.0, 0.0), (1.0, 0.0)),
            ((0.0, 0.5), (1.0, 0.5)),
            ((0.0, 1.0), (1.0, 1.0)),
        ],
        mirror_pairs_v =[
            ((0.0, 0.0), (0.0, 1.0)),
            ((0.5, 0.0), (0.5, 1.0)),
            ((1.0, 0.0), (1.0, 1.0)),
        ],
        mirror_pairs_mid =[
            ((0.25, 0.0), (0.25, 1.0)),
            ((0.75, 0.0), (0.75, 1.0)),
        ],
    ),
    'oblique': PlaneGroupPattern(
        corner_edge_center_pts = [(0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
                                  (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
                                  (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)],
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

class MixinPGSymmetry:
    def show(self, ax=None, alpha=0.5):
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.2, 7.2))

        ax.imshow(self.img, cmap = 'gray')
        ax.scatter(self.unit_cell_corners[:, 0], self.unit_cell_corners[:, 1], s=10, color='red', zorder=5)
        cell = self.unit_cell

        try:
            pattern = PG_PATTERNS[self.pg_number]
        except KeyError:
            raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

        P2 = transform_via_cell(pattern.P2, cell)
        P3 = transform_via_cell(pattern.P3, cell)
        P4 = transform_via_cell(pattern.P4, cell)
        P6 = transform_via_cell(pattern.P6, cell)

        mirror_pairs = transform_via_cell(pattern.mirror_pairs, cell)
        glide_pairs = transform_via_cell(pattern.glide_pairs, cell)
        outline_pairs = transform_via_cell(pattern.outline_pairs, cell)

        if P2 is not None:
            P2 += self.unit_cell_corners[0]
            add_rotational_centers(ax, P2, n_fold=2, color='C0', zorder=5, s=100, alpha=alpha, label='2-fold')
        if P3 is not None:
            P3 += self.unit_cell_corners[0]
            add_rotational_centers(ax, P3, n_fold=3, color='C1', zorder=5, s=100, alpha=alpha, label='3-fold')
        if P4 is not None:
            P4 += self.unit_cell_corners[0]
            add_rotational_centers(ax, P4, n_fold=4, color='C2', zorder=5, s=100, alpha=alpha, label='4-fold')
        if P6 is not None:
            P6 += self.unit_cell_corners[0]
            add_rotational_centers(ax, P6, n_fold=6, color='C3', zorder=5, s=100, alpha=alpha, label='6-fold')

        if mirror_pairs is not None:
            mirror_pairs += self.unit_cell_corners[0]
            add_lines(ax, mirror_pairs, lw=2, color='cyan', alpha=alpha, label='mirror line')

        if glide_pairs is not None:
            glide_pairs += self.unit_cell_corners[0]
            add_lines(ax, glide_pairs, lw=1, ls='--', color='yellow', alpha=alpha, label='glide line')

        outline_pairs += self.unit_cell_corners[0]
        add_lines(ax, outline_pairs, lw=0.5, color='red', alpha=alpha, label='outline')

        l = 1.5 * np.maximum(np.ptp(outline_pairs[:, 0]), np.ptp(outline_pairs[:, 1]))
        x0 = np.mean(outline_pairs[:, 0])
        y0 = np.mean(outline_pairs[:, 1])

        ax.set_xlim(x0 - l, x0 + l)
        ax.set_ylim(y0 - l, y0 + l)
        #ax.axis('equal')
        ax.legend(ncol=2)
        ax.axis('off')


class MixinPGFeatures:

    def get_features(self):
        if self.pg_number in [13, 14, 15, 16, 17]:
            return _get_features_pg_hexagonal(self)
        elif self.pg_number in [10, 11, 12]:
            return _get_features_pg_square(self)
        elif self.pg_number in [3, 4, 5, 6, 7, 8, 9]:
            return _get_features_pg_rectangle(self)
        elif self.pg_number in [1, 2]:
            return _get_features_pg_oblique(self)
        else:
            raise NotImplementedError(f"No features defined for pg {self.pg_number}")


    def is_typical(self, verbose=False):
        if self.pg_number == 1:
            return _is_typical_pg1(self, verbose=verbose)
        elif self.pg_number == 2:
            return _is_typical_pg2(self, verbose=verbose)
        elif self.pg_number == 3:
            return _is_typical_pg3(self, verbose=verbose)
        elif self.pg_number == 4:
            return _is_typical_pg4(self, verbose=verbose)
        elif self.pg_number == 5:
            return _is_typical_pg5(self, verbose=verbose)
        elif self.pg_number == 6:
            return _is_typical_pg6(self, verbose=verbose)
        elif self.pg_number == 7:
            return _is_typical_pg7(self, verbose=verbose)
        elif self.pg_number == 8:
            return _is_typical_pg8(self, verbose=verbose)
        elif self.pg_number == 9:
            return _is_typical_pg9(self, verbose=verbose)
        elif self.pg_number == 10:
            return _is_typical_pg10(self, verbose=verbose)
        elif self.pg_number == 11:
            return _is_typical_pg11(self, verbose=verbose)
        elif self.pg_number == 12:
            return _is_typical_pg12(self, verbose=verbose)
        elif self.pg_number == 13:
            return _is_typical_pg13(self, verbose=verbose)
        elif self.pg_number == 14:
            return _is_typical_pg14(self, verbose=verbose)
        elif self.pg_number == 15:
            return _is_typical_pg15(self, verbose=verbose)
        elif self.pg_number == 16:
            return _is_typical_pg16(self, verbose=verbose)
        elif self.pg_number == 17:
            return _is_typical_pg17(self, verbose=verbose)
        else:
            raise NotImplementedError(f"No features defined for pg {self.pg_number}")


def _get_features_pg_hexagonal(pgimage):
    cell = pgimage.unit_cell
    if not pgimage.has_symm_maps:
        pgimage.compute_symm_maps(patch_size=None, n_max=20)
    try:
        pattern = PG_PATTERNS['hexagonal']
    except KeyError:
        raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

    # transform all coordinate sets through the cell
    pts1 = transform_via_cell(pattern.corner_pts, cell) + pgimage.unit_cell_corners[0]
    pts2 = transform_via_cell(pattern.hexagonal_inside_pts, cell) + pgimage.unit_cell_corners[0]
    pts3 = transform_via_cell(pattern.hexagonal_edge_pts, cell) + pgimage.unit_cell_corners[0]

    mirror_pairs1  = transform_via_cell(pattern.mirror_pairs1, cell) + pgimage.unit_cell_corners[0]
    mirror_pairs2  = transform_via_cell(pattern.mirror_pairs2, cell) + pgimage.unit_cell_corners[0]


    P1_I2 = _get_intensity(pts1, pgimage.rot_maps[0])
    P1_I3 = _get_intensity(pts1, pgimage.rot_maps[1])
    P1_I4 = _get_intensity(pts1, pgimage.rot_maps[2])
    P1_I6 = _get_intensity(pts1, pgimage.rot_maps[3])

    P2_I2 = _get_intensity(pts2, pgimage.rot_maps[0])
    P2_I3 = _get_intensity(pts2, pgimage.rot_maps[1])
    P2_I4 = _get_intensity(pts2, pgimage.rot_maps[2])
    P2_I6 = _get_intensity(pts2, pgimage.rot_maps[3])

    P3_I2 = _get_intensity(pts3, pgimage.rot_maps[0])
    P3_I3 = _get_intensity(pts3, pgimage.rot_maps[1])
    P3_I4 = _get_intensity(pts3, pgimage.rot_maps[2])
    P3_I6 = _get_intensity(pts3, pgimage.rot_maps[3])

    lines1_mean = []
    lines1_std = []

    for (pt1, pt2) in mirror_pairs1:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines1_mean.append(np.mean(intensity))
        lines1_std.append(np.std(intensity))
    lines1_feat = np.hstack([lines1_mean,lines1_std])

    lines2_mean = []
    lines2_std = []
    for (pt1, pt2) in mirror_pairs2:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines2_mean.append(np.mean(intensity))
        lines2_std.append(np.std(intensity))
    lines2_feat = np.hstack([lines2_mean,lines2_std])

    return np.hstack([P1_I2, P1_I3, P1_I4, P1_I6,
                      P2_I2, P2_I3, P2_I4, P2_I6,
                      P3_I2, P3_I3, P3_I4, P3_I6,
                      lines1_feat, lines2_feat])

def _get_features_pg_square(pgimage):
    cell = pgimage.unit_cell
    if not pgimage.has_symm_maps:
        pgimage.compute_symm_maps(patch_size=None, n_max=20)
    try:
        pattern = PG_PATTERNS['square']
    except KeyError:
        raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

    # transform all coordinate sets through the cell
    pts1 = transform_via_cell(pattern.square_edge_pts, cell) + pgimage.unit_cell_corners[0]
    pts2 = transform_via_cell(pattern.corner_center_pts, cell) + pgimage.unit_cell_corners[0]

    mirror_pairs1  = transform_via_cell(pattern.mirror_pairs1, cell) + pgimage.unit_cell_corners[0]
    mirror_pairs2  = transform_via_cell(pattern.mirror_pairs2, cell) + pgimage.unit_cell_corners[0]


    P1_I2 = _get_intensity(pts1, pgimage.rot_maps[0])
    P1_I3 = _get_intensity(pts1, pgimage.rot_maps[1])
    P1_I4 = _get_intensity(pts1, pgimage.rot_maps[2])
    P1_I6 = _get_intensity(pts1, pgimage.rot_maps[3])

    P2_I2 = _get_intensity(pts2, pgimage.rot_maps[0])
    P2_I3 = _get_intensity(pts2, pgimage.rot_maps[1])
    P2_I4 = _get_intensity(pts2, pgimage.rot_maps[2])
    P2_I6 = _get_intensity(pts2, pgimage.rot_maps[3])

    lines1_mean = []
    lines1_std = []

    for (pt1, pt2) in mirror_pairs1:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines1_mean.append(np.mean(intensity))
        lines1_std.append(np.std(intensity))
    lines1_feat = np.hstack([lines1_mean,lines1_std])

    lines2_mean = []
    lines2_std = []
    for (pt1, pt2) in mirror_pairs2:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines2_mean.append(np.mean(intensity))
        lines2_std.append(np.std(intensity))
    lines2_feat = np.hstack([lines2_mean,lines2_std])

    return np.hstack([P1_I2, P1_I3, P1_I4, P1_I6,
                      P2_I2, P2_I3, P2_I4, P2_I6,
                      lines1_feat, lines2_feat])

def _get_features_pg_oblique(pgimage):
    cell = pgimage.unit_cell
    if not pgimage.has_symm_maps:
        pgimage.compute_symm_maps(patch_size=None, n_max=20)
    try:
        pattern = PG_PATTERNS['oblique']
    except KeyError:
        raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

    # transform all coordinate sets through the cell
    pts1 = transform_via_cell(pattern.corner_edge_center_pts, cell) + pgimage.unit_cell_corners[0]

    P1_I2 = _get_intensity(pts1, pgimage.rot_maps[0])
    P1_I3 = _get_intensity(pts1, pgimage.rot_maps[1])
    P1_I4 = _get_intensity(pts1, pgimage.rot_maps[2])
    P1_I6 = _get_intensity(pts1, pgimage.rot_maps[3])

    return np.hstack([P1_I2, P1_I3, P1_I4, P1_I6])

def _get_features_pg_rectangle(pgimage):
    cell = pgimage.unit_cell
    if not pgimage.has_symm_maps:
        pgimage.compute_symm_maps(patch_size=None, n_max=20)
    try:
        pattern = PG_PATTERNS['rect']
    except KeyError:
        raise NotImplementedError(f"No pattern defined for pg {self.pg_number}")

    # transform all coordinate sets through the cell
    pts1 = transform_via_cell(pattern.corner_edge_center_pts, cell) + pgimage.unit_cell_corners[0]
    pts2 = transform_via_cell(pattern.rect_center_pts, cell) + pgimage.unit_cell_corners[0]

    mirror_pairs1  = transform_via_cell(pattern.mirror_pairs_h, cell) + pgimage.unit_cell_corners[0]
    mirror_pairs2  = transform_via_cell(pattern.mirror_pairs_v, cell) + pgimage.unit_cell_corners[0]
    mirror_pairs3 = transform_via_cell(pattern.mirror_pairs_mid, cell) + pgimage.unit_cell_corners[0]


    P1_I2 = _get_intensity(pts1, pgimage.rot_maps[0])
    P1_I3 = _get_intensity(pts1, pgimage.rot_maps[1])
    P1_I4 = _get_intensity(pts1, pgimage.rot_maps[2])
    P1_I6 = _get_intensity(pts1, pgimage.rot_maps[3])

    P2_I2 = _get_intensity(pts2, pgimage.rot_maps[0])
    P2_I3 = _get_intensity(pts2, pgimage.rot_maps[1])
    P2_I4 = _get_intensity(pts2, pgimage.rot_maps[2])
    P2_I6 = _get_intensity(pts2, pgimage.rot_maps[3])

    lines1_mean = []
    lines1_std = []

    for (pt1, pt2) in mirror_pairs1:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines1_mean.append(np.mean(intensity))
        lines1_std.append(np.std(intensity))
    lines1_feat = np.hstack([lines1_mean,lines1_std])

    lines2_mean = []
    lines2_std = []
    for (pt1, pt2) in mirror_pairs2:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines2_mean.append(np.mean(intensity))
        lines2_std.append(np.std(intensity))
    lines2_feat = np.hstack([lines2_mean,lines2_std])

    lines3_mean = []
    lines3_std = []
    for (pt1, pt2) in mirror_pairs3:
        line = get_line(pt1, pt2)
        intensity = _get_intensity(line, pgimage.ref_map)
        lines3_mean.append(np.mean(intensity))
        lines3_std.append(np.std(intensity))
    lines3_feat = np.hstack([lines3_mean, lines3_std])

    return np.hstack([P1_I2, P1_I3, P1_I4, P1_I6,
                      P2_I2, P2_I3, P2_I4, P2_I6,
                      lines1_feat, lines2_feat, lines3_feat])


def _get_intensity(pts_array, data):
    if pts_array.ndim == 2:
        pts_xy = np.round(pts_array).astype(int)
        return np.array([data[y, x] for (x, y) in pts_xy])
    else:
        raise ValueError('pts_array must have a dimension of 2.')


def is_mirror(arr,t1 = 0.8, t2 = 0.7):
    arr = np.array(arr)
    if len(arr) % 2 != 0:
        raise ValueError("length must be even.")

    half_len = len(arr) // 2
    means = arr[:half_len]
    stds = arr[half_len:]

    differences = means - stds

    floor = np.all(differences > t1)
    ceiling = np.all(differences < t2)

    if floor:
        return 1
    elif ceiling:
        return 0
    else:
        return None

def is_rot2(arr,t1 = 0.85, t2 = 0.7):
    arr = np.array(arr)
    if len(arr) % 4 != 0:
        raise ValueError("length must 4n.")
    n = len(arr) // 4
    rot2,rot3,rot4,rot6 = arr.reshape(4, n)
    if np.mean(rot2)<t1 or np.mean(rot3)>t2 or np.mean(rot4)>t2 or np.mean(rot6)>t2:
        return False
    else:
        return True

def is_rot3(arr,t1 = 0.85, t2 = 0.7):
    arr = np.array(arr)
    if len(arr) % 4 != 0:
        raise ValueError("length must 4n.")
    n = len(arr) // 4
    rot2,rot3,rot4,rot6 = arr.reshape(4, n)
    if np.mean(rot3)<t1 or np.mean(rot2)>t2 or np.mean(rot4)>t2 or np.mean(rot6)>t2:
        return False
    else:
        return True

def is_rot4(arr,t1 = 0.85, t2 = 0.7):
    arr = np.array(arr)
    if len(arr) % 4 != 0:
        raise ValueError("length must 4n.")
    n = len(arr) // 4
    rot2,rot3,rot4,rot6 = arr.reshape(4, n)
    if np.mean(rot4)<t1 or np.mean(rot3)>t2 or np.mean(rot6)>t2:
        return False
    else:
        return True

def is_rot6(arr,t1 = 0.85, t2 = 0.7):
    arr = np.array(arr)
    if len(arr) % 4 != 0:
        raise ValueError("length must 4n.")
    n = len(arr) // 4
    rot2,rot3,rot4,rot6 = arr.reshape(4, n)
    if np.mean(rot6)<t1 or np.mean(rot4)>t2:
        return False
    else:
        return True

def _is_typical_pg1(pgimage, verbose = False):
    X = _get_features_pg_oblique(pgimage)
    corner_edge_center_pts = X

    conner_edge_center_rot2_status = is_rot2(corner_edge_center_pts)
    corner_edge_center_rot3_status = is_rot3(corner_edge_center_pts)

    stauts = (conner_edge_center_rot2_status, corner_edge_center_rot3_status)

    if stauts == (0,0):
        return True
    else:
        return False


def _is_typical_pg2(pgimage, verbose = False):
    X = _get_features_pg_oblique(pgimage)
    corner_edge_center_pts = X

    conner_edge_center_rot2_status = is_rot2(corner_edge_center_pts)

    stauts = (conner_edge_center_rot2_status)

    if stauts == (1):
        return True
    else:
        return False


def _is_typical_pg3(pgimage, verbose = False):
    X = _get_features_pg_rectangle(pgimage)
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (0, 1, 0):
        return True
    else:
        return False


def _is_typical_pg4(pgimage, verbose = False):
    X = _get_features_pg_rectangle(pgimage)
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (0, 0, 0):
        return True
    else:
        return False

def _is_typical_pg5(pgimage, verbose = False):
    X = _get_features_pg_rectangle(pgimage)
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (0, 1, 0):
        return True
    else:
        return False

def _is_typical_pg6(pgimage, verbose = False):
    X = _get_features_pg_rectangle(pgimage)
    corner_center_pts_2_fold = X[0:9 * 4]
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    corner_center_rot2_status = is_rot2(corner_center_pts_2_fold)
    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (corner_center_rot2_status, h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (1, 1, 1, 0):
        return True
    else:
        return False

def _is_typical_pg7(pgimage, verbose = False):
    X = _get_features_pg_rectangle(pgimage)
    corner_center_pts_2_fold = X[0:9 * 4]
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    corner_center_rot2_status = is_rot2(corner_center_pts_2_fold)
    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (corner_center_rot2_status, h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (1, 0, 0, 1):
        return True
    else:
        return False

def _is_typical_pg8(pgimage, verbose = False):
    X = _get_features_pg_rectangle(pgimage)
    corner_center_pts_2_fold = X[0:9 * 4]
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    corner_center_rot2_status = is_rot2(corner_center_pts_2_fold)
    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (corner_center_rot2_status, h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (1, 0, 0, 0):
        return True
    else:
        return False

def _is_typical_pg9(pgimage, verbose=False):
    X = _get_features_pg_rectangle(pgimage)
    corner_center_pts_2_fold = X[0:9 * 4]
    rect_center_pts_2_fold = X[36:36 + 16]
    h_lines = X[52:52 + 6]
    v_lines = X[58:58 + 6]
    mid_lines = X[64:]

    corner_center_rot2_status = is_rot2(corner_center_pts_2_fold)
    rect_center_rot2_status = is_rot2(rect_center_pts_2_fold)
    h_mirror_status = is_mirror(h_lines)
    v_mirror_status = is_mirror(v_lines)
    mid_mirror_status = is_mirror(mid_lines)

    status = (corner_center_rot2_status, rect_center_rot2_status, h_mirror_status, v_mirror_status, mid_mirror_status)

    if status == (1,1,1,1,0):
        return True
    else:
        return False


def _is_typical_pg10(pgimage, verbose=False):
    X = _get_features_pg_square(pgimage)
    square_edge_pts_2_fold = X[0:16]
    corner_center_pts_4_fold = X[16:16+20]
    pg11_mirror_line = X[36:36+9*2]
    pg12_mirror_line = X[54:]

    square_edge_rot_2_status = is_rot2(square_edge_pts_2_fold)
    corner_center_rot_4_status = is_rot4(corner_center_pts_4_fold)
    pg11_mirror_status = is_mirror(pg11_mirror_line)
    pg12_mirror_status = is_mirror(pg12_mirror_line)

    status = (square_edge_rot_2_status, corner_center_rot_4_status, pg11_mirror_status, pg12_mirror_status)

    if status == (1,1,0,0):
        return True
    else:
        return False

def _is_typical_pg11(pgimage, verbose=False):
    X = _get_features_pg_square(pgimage)
    square_edge_pts_2_fold = X[0:16]
    corner_center_pts_4_fold = X[16:16 + 20]
    pg11_mirror_line = X[36:36 + 9 * 2]
    pg12_mirror_line = X[54:]

    square_edge_rot_2_status = is_rot2(square_edge_pts_2_fold)
    corner_center_rot_4_status = is_rot4(corner_center_pts_4_fold)
    pg11_mirror_status = is_mirror(pg11_mirror_line)
    pg12_mirror_status = is_mirror(pg12_mirror_line)

    status = (square_edge_rot_2_status, corner_center_rot_4_status, pg11_mirror_status, pg12_mirror_status)

    if status == (1,1,1,0):
        return True
    else:
        return False

def _is_typical_pg12(pgimage, verbose=False):
    X = _get_features_pg_square(pgimage)
    square_edge_pts_2_fold = X[0:16]
    corner_center_pts_4_fold = X[16:16 + 20]
    pg11_mirror_line = X[36:36 + 9 * 2]
    pg12_mirror_line = X[54:]

    square_edge_rot_2_status = is_rot2(square_edge_pts_2_fold)
    corner_center_rot_4_status = is_rot4(corner_center_pts_4_fold)
    pg11_mirror_status = is_mirror(pg11_mirror_line)
    pg12_mirror_status = is_mirror(pg12_mirror_line)

    status = (square_edge_rot_2_status, corner_center_rot_4_status, pg11_mirror_status, pg12_mirror_status)

    if status == (1,1,0,1):
        return True
    else:
        return False

def _is_typical_pg13(pgimage, verbose=False):
    X = _get_features_pg_hexagonal(pgimage)
    hexa_corner_pts_3_fold = X[0:16]
    hexa_inside_pts_3_fold = X[16:16+8]
    pg_14_mirror_line = X[44:54]
    pg_15_mirror_line = X[54:]

    corner_rot_3_status = is_rot3(hexa_corner_pts_3_fold)
    inside_rot_3_status = is_rot3(hexa_inside_pts_3_fold)
    pg_14_mirror_status = is_mirror(pg_14_mirror_line)
    pg_15_mirror_status = is_mirror(pg_15_mirror_line)

    status = (corner_rot_3_status, inside_rot_3_status, pg_14_mirror_status, pg_15_mirror_status)

    if status == (1,1,0,0):
        return True
    else:
        return False



def _is_typical_pg14(pgimage, verbose=False):
    X = _get_features_pg_hexagonal(pgimage)
    hexa_corner_pts_3_fold = X[0:16]
    hexa_inside_pts_3_fold = X[16:16 + 8]
    pg_14_mirror_line = X[44:54]
    pg_15_mirror_line = X[54:]

    corner_rot_3_status = is_rot3(hexa_corner_pts_3_fold)
    inside_rot_3_status = is_rot3(hexa_inside_pts_3_fold)
    pg_14_mirror_status = is_mirror(pg_14_mirror_line)
    pg_15_mirror_status = is_mirror(pg_15_mirror_line)

    status = (corner_rot_3_status, inside_rot_3_status, pg_14_mirror_status, pg_15_mirror_status)

    if status == (1, 1, 1, 0):
        return True
    else:
        return False


def _is_typical_pg15(pgimage, verbose=False):
    X = _get_features_pg_hexagonal(pgimage)
    hexa_corner_pts_3_fold = X[0:16]
    hexa_inside_pts_3_fold = X[16:16 + 8]
    pg_14_mirror_line = X[44:54]
    pg_15_mirror_line = X[54:]

    corner_rot_3_status = is_rot3(hexa_corner_pts_3_fold)
    inside_rot_3_status = is_rot3(hexa_inside_pts_3_fold)
    pg_14_mirror_status = is_mirror(pg_14_mirror_line)
    pg_15_mirror_status = is_mirror(pg_15_mirror_line)

    status = (corner_rot_3_status, inside_rot_3_status, pg_14_mirror_status, pg_15_mirror_status)

    if status == (1, 1, 0, 1):
        return True
    else:
        return False

def _is_typical_pg16(pgimage, verbose=False):
    X = _get_features_pg_hexagonal(pgimage)
    hexa_corner_pts_6_fold = X[0:16]
    hexa_inside_pts_3_fold = X[16:16 + 8]
    hexa_edge_pts_2_fold = X[24:24+20]
    pg_14_mirror_line = X[44:54]
    pg_15_mirror_line = X[54:]

    corner_rot_6_status = is_rot6(hexa_corner_pts_6_fold)
    inside_rot_3_status = is_rot3(hexa_inside_pts_3_fold)
    edge_pts_2_status = is_rot2(hexa_edge_pts_2_fold)
    pg_14_mirror_status = is_mirror(pg_14_mirror_line)
    pg_15_mirror_status = is_mirror(pg_15_mirror_line)

    status = (corner_rot_6_status, inside_rot_3_status, edge_pts_2_status, pg_14_mirror_status, pg_15_mirror_status)

    if status == (1, 1, 1, 0, 0):
        return True
    else:
        return False

def _is_typical_pg17(pgimage, verbose=False):
    X = _get_features_pg_hexagonal(pgimage)
    hexa_corner_pts_6_fold = X[0:16]
    hexa_inside_pts_3_fold = X[16:16 + 8]
    hexa_edge_pts_2_fold = X[24:24 + 20]
    pg_14_mirror_line = X[44:54]
    pg_15_mirror_line = X[54:]

    corner_rot_6_status = is_rot6(hexa_corner_pts_6_fold)
    inside_rot_3_status = is_rot3(hexa_inside_pts_3_fold)
    edge_pts_2_status = is_rot2(hexa_edge_pts_2_fold)
    pg_14_mirror_status = is_mirror(pg_14_mirror_line)
    pg_15_mirror_status = is_mirror(pg_15_mirror_line)

    status = (corner_rot_6_status, inside_rot_3_status, edge_pts_2_status, pg_14_mirror_status, pg_15_mirror_status)

    if status == (1, 1, 1, 1, 1):
        return True
    else:
        return False
