try:
    from spglib import get_symmetry_layerdataset  # spglib >= 2.7 (pybind11)
except ImportError:
    from spglib.spglib import get_symmetry_layerdataset  # spglib < 2.7 (ctypes)

from ._utils import layer2plane

def get_layer_group(atoms, aperiodic_axis=2, symprec=1e-5):
    """
    Determine the layer group of a 2D slab stored as an ASE Atoms.

    Parameters
    ----------
    atoms : ase.Atoms
        Your slab, with enough vacuum along the non-periodic direction.
    aperiodic_axis : int, optional
        Which axis to treat as aperiodic (0 → a, 1 → b, 2 → c). Default is 2 (Z).
    symprec : float, optional
        Distance tolerance for symmetry matching. Default is 1e-3.

    Returns
    -------
    layergroup_number : int
        The layer group number (1–80).
    layergroup_symbol : str
        The international Hermann–Mauguin symbol (e.g. “p-3m1”).
    """
    # 1) pack ASE cell into spglib format
    lattice     = atoms.get_cell().array
    positions   = atoms.get_scaled_positions()
    numbers     = atoms.get_atomic_numbers()
    cell        = (lattice, positions, numbers)

    # 2) search for layer group
    ds = get_symmetry_layerdataset(cell,
                                   aperiodic_dir=aperiodic_axis,
                                   symprec=symprec)
    if ds is None:
        raise RuntimeError(
            "Layer-group search failed – try more vacuum or a larger symprec."
        )

    # 3) return symmetry dataset
    return ds

def get_plane_group(atoms, aperiodic_axis=2, symprec=1e-5):
    ds = get_layer_group(atoms, aperiodic_axis=aperiodic_axis, symprec=symprec)
    lg_number = ds.number
    try:
        pg_number = layer2plane(lg_number)
    except KeyError:
        raise ValueError(f"No mapping to a plane group for layer group #{lg_number}")
    return pg_number

