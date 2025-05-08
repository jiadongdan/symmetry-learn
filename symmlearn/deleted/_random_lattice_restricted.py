import numpy as np
from ase import Atoms
from skimage.filters import gaussian
from pyxtal import pyxtal

from symmlearn.lattice._utils import rotate_atoms_xy_center, crop_atoms_xy_center

# code adapted from: https://github.com/jarek-pawlowski/wallpaper_group/blob/main/utils_gen.py

plane_to_layer_groups = {
    1:  [1, 4, 5],
    2:  [2, 3, 6, 7],
    3:  [8, 11, 27, 28, 36],
    4:  [9, 12, 29, 32, 33],
    5:  [10, 13, 34, 35],
    6:  [14, 19, 23, 37, 38, 41, 48],
    7:  [15, 16, 20, 24, 40, 43, 45],
    8:  [17, 21, 25, 44],
    9:  [18, 22, 26, 42, 47],
    10: [39, 46, 49, 50, 51],
    11: [53, 55, 57, 59, 61, 62, 64],
    12: [52, 54, 56, 58, 60, 63],
    13: [65, 66, 74],
    14: [67, 69],
    15: [68, 70],
    16: [66, 73, 75],
    17: [71, 72, 76, 77]
}

# mapping in original GitHub repo is wrong.
# "name" = wallpaper group name,
# "layer_num" = layer group number,
# "num_atoms" = bounds for number of atoms in unit cell,
# "even" = shoud number of atoms in unit cell be even? some layer group only accept even number atoms
wallpaper_groups = {1:  {"name" : "p1",   "layer_num" : 4,  "num_atoms" : [1, 10], "even" : False},  # 1, 4, 5
                    2:  {"name" : "p2",   "layer_num" : 3,  "num_atoms" : [3, 10], "even" : False},  # 2, 3, 6, 7
                    3:  {"name" : "pm",   "layer_num" : 27, "num_atoms" : [3, 10], "even" : True},   # 8, 11, 27, 28, 36
                    4:  {"name" : "pg",   "layer_num" : 29, "num_atoms" : [4, 10], "even" : True},   # 9, 12, 29, 32, 33
                    5:  {"name" : "pmm",  "layer_num" : 35, "num_atoms" : [4, 16], "even" : True},   # 10, 13, 34, 35
                    6:  {"name" : "pmg",  "layer_num" : 23, "num_atoms" : [4, 16], "even" : True},   # 14, 19, 23, 37, 38, 41, 48
                    7:  {"name" : "pgg",  "layer_num" : 24, "num_atoms" : [4, 16], "even" : True},   # 15, 16, 20, 24, 40, 43, 45
                    8:  {"name" : "cm",   "layer_num" : 21, "num_atoms" : [4, 16], "even" : True},   # 17, 21, 25, 44
                    9:  {"name" : "cmm",  "layer_num" : 26, "num_atoms" : [4, 16], "even" : True},   # 18, 22, 26, 42, 47
                    10: {"name" : "p4",   "layer_num" : 49, "num_atoms" : [4, 20], "even" : True},   # 39, 46, 49, 50, 51
                    11: {"name" : "p4m",  "layer_num" : 55, "num_atoms" : [4, 20], "even" : True},   # 53, 55, 57, 59, 61, 62, 64
                    12: {"name" : "p4g",  "layer_num" : 56, "num_atoms" : [4, 20], "even" : True},   # 52, 54, 56, 58, 60, 63
                    13: {"name" : "p3",   "layer_num" : 65, "num_atoms" : [4, 20], "even" : True},   # 65, 66, 74
                    14: {"name" : "p3m1", "layer_num" : 69, "num_atoms" : [4, 20], "even" : True},   # 67, 69
                    15: {"name" : "p31m", "layer_num" : 70, "num_atoms" : [4, 20], "even" : True},   # 68, 70
                    16: {"name" : "p6",   "layer_num" : 73, "num_atoms" : [6, 20], "even" : True},   # 66, 73, 75
                    17: {"name" : "p6m",  "layer_num" : 77, "num_atoms" : [6, 20], "even" : True}}   # 71, 72, 76, 77

def make_cell_rectangular(atoms: Atoms):
    new_atoms = atoms.copy()
    cell = new_atoms.cell.copy()

    # in xy plane
    cell[0, 1] = 0  # Remove xy shear
    cell[1, 0] = 0  # Remove yx shear

    new_atoms.set_cell(cell, scale_atoms=False) # scale_atoms=False, Prevent atomic position scaling
    new_atoms.wrap()  # Ensure atoms are inside the new cell
    return new_atoms

def make_cell_square(atoms: Atoms):
    new_atoms = make_cell_rectangular(atoms)
    cell = new_atoms.cell.copy()

    # Create the new square cell
    size = min(cell[0, 0], cell[1, 1])
    cell[0, 0] = size
    cell[1, 1] = size

    # Before set_cell, we have to make periodic boundary condition False,
    # then scaled_positions won't be scaled to [0, 1]
    new_atoms.set_pbc(False)

    # Apply the new cell
    new_atoms.set_cell(cell, scale_atoms=False)  # positions do not change, but scaled_positions updated
    scaled_positions = new_atoms.get_scaled_positions()
    # Remove atoms that are outside [0,1) in any direction
    mask = (scaled_positions >= 0).all(axis=1) & (scaled_positions < 1).all(axis=1)
    new_atoms_ = new_atoms[mask]  # Remove atoms outside the new cell

    return new_atoms_

def random_supercell(size_min, size_max, rng):
    size = rng.integers(size_min, size_max, endpoint=True)
    return [size, size, 1]

def has_duplicate_xy(points: np.ndarray) -> bool:
    """
    Vectorized check for duplicate (x, y) pairs.
    """
    # take only x,y columns
    xy = points[:, :2]
    # convert each row to a single bytes object, then count uniques
    dtype = np.dtype((np.void, xy.dtype.itemsize * xy.shape[1]))
    packed = np.ascontiguousarray(xy).view(dtype)
    _, counts = np.unique(packed, return_counts=True)
    return np.any(counts > 1)

class RandomLattice:

    def __init__(self, size_min=15, size_max=20, atom_radius=.3, scale=1.):

        self.size_min = size_min
        self.size_max = size_max
        self.atom_radius = atom_radius
        self.scale = scale
        self.group = None
        self.supercell = None
        self.atoms = None
        self.atoms_unit_cell = None
        self.atoms_ase = None

    def generate_lattice(self, wallpaper_class, num_atoms=None, seed=None, debug=False):
        # Create a SeedSequence from the main seed
        seed_seq = np.random.SeedSequence(seed)

        # Generate independent child seeds
        child_seeds = seed_seq.spawn(4)

        # Create independent random generators
        rng1 = np.random.default_rng(child_seeds[0]) # this is for num_atoms_in_unit_cell
        rng2 = np.random.default_rng(child_seeds[1]) # this is for construct random structure
        rng3 = np.random.default_rng(child_seeds[2]) # this is for rotation
        rng4 = np.random.default_rng(child_seeds[2]) # this is for supercell

        # get a random angle
        angle_deg = rng3.uniform(0, 360)

        # get random supercell size
        self.supercell = random_supercell(self.size_min, self.size_max, rng4)

        if wallpaper_class not in range(1, 18):
            raise ValueError("wallpaper_class should be between 1 and 17")
        self.group = wallpaper_groups[wallpaper_class]

        # define random crystal
        struct = pyxtal()
        _ = struct.from_random(dim=2, group=self.group["layer_num"],
                               species=['C'], numIons=[num_atoms],
                               thickness=6., random_state=rng2)
        self.atoms_unit_cell = struct.to_ase()

        while has_duplicate_xy(self.atoms_unit_cell.get_positions()):
            bitgen2 = rng2.bit_generator.jumped()
            rng2 = np.random.Generator(bitgen2)
            _ = struct.from_random(dim=2, group=self.group["layer_num"],
                                   species=['C'], numIons=[num_atoms],
                                   thickness=6., random_state=rng2)
            self.atoms_unit_cell = struct.to_ase()

        self.ase_atoms = self.atoms_unit_cell * self.supercell
        self.atoms = make_cell_square(self.ase_atoms)
        self.atoms = rotate_atoms_xy_center(self.atoms, angle_deg)
        self.atoms = crop_atoms_xy_center(self.atoms)
        return self.atoms

    def generate_data(self, ase_atoms, sigma=5, size=512):
        a = ase_atoms.cell.cellpar()[0]
        xyz = ase_atoms.get_positions() * (size - 1) / a
        x = np.round(xyz[:, 0]).astype(int)
        y = np.round(xyz[:, 1]).astype(int)
        shape = (size, size)
        array = np.zeros(shape)
        array[x, y] = 1.
        array = gaussian(array, sigma=sigma, mode='constant')
        return array