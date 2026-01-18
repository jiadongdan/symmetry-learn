import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
import numpy as np

def show_atoms(atoms, direction='xy', ax=None):
    """
    Visualize an ASE Atoms object projected onto a 2D plane.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure to visualize.
    direction : str, optional
        The viewing direction/projection plane. Options:
        - 'xy': View along z-axis (default)
        - 'xz': View along y-axis
        - 'yz': View along x-axis
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    positions = atoms.get_positions()
    atomic_numbers = atoms.get_atomic_numbers()
    cell = atoms.get_cell()

    # Select coordinates based on direction
    if direction == 'xy':
        x_idx, y_idx = 0, 1
        xlabel, ylabel = 'x (Å)', 'y (Å)'
        cell_vectors = cell[[0, 1], :2]
    elif direction == 'xz':
        x_idx, y_idx = 0, 2
        xlabel, ylabel = 'x (Å)', 'z (Å)'
        cell_vectors = cell[[0, 2]][:, [0, 2]]
    elif direction == 'yz':
        x_idx, y_idx = 1, 2
        xlabel, ylabel = 'y (Å)', 'z (Å)'
        cell_vectors = cell[[1, 2]][:, [1, 2]]
    else:
        raise ValueError(f"Invalid direction '{direction}'. Choose from 'xy', 'xz', 'yz'.")

    x = positions[:, x_idx]
    y = positions[:, y_idx]

    # Sort by depth (the remaining axis) so closer atoms are drawn on top
    depth_idx = {'xy': 2, 'xz': 1, 'yz': 0}[direction]
    depth = positions[:, depth_idx]
    sort_idx = np.argsort(depth)

    # Vectorized: get all properties at once
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    z_sorted = atomic_numbers[sort_idx]

    colors = jmol_colors[z_sorted]
    radii = covalent_radii[z_sorted] * 0.5
    diameters = radii * 2

    # Create EllipseCollection (circles are ellipses with equal width/height)
    # units='x' means sizes are in data coordinates
    ec = EllipseCollection(
        widths=diameters,
        heights=diameters,
        angles=0,
        units='x',
        offsets=np.column_stack([x_sorted, y_sorted]),
        transOffset=ax.transData,
        facecolors=colors,
        edgecolors='black',
        linewidths=0.5,
    )
    ax.add_collection(ec)

    # Draw unit cell outline
    if cell.any():
        origin = np.array([0, 0])
        v1 = cell_vectors[0]
        v2 = cell_vectors[1]
        cell_corners = np.array([
            origin, origin + v1, origin + v1 + v2, origin + v2, origin
        ])
        ax.plot(cell_corners[:, 0], cell_corners[:, 1], 'k-', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect('equal')
    ax.autoscale_view()

    return ax