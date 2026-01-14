import numpy as np
from ._wyckoff_position import wyckoff_pos, WyckoffPosition
from ._plane_group import PlaneGroup
from ._layer_group import get_plane_group


class WyckoffStructure:

    def __init__(self, pg_num, structure_letters):
        self.structure_letters = structure_letters
        self.pg_num = pg_num
        self.unique_atoms = len(structure_letters)

    @property
    def num_atoms(self):
        return sum([len(wyckoff_pos[self.pg_num][letter]) for letters in self.structure_letters for letter in letters])

    def is_all_fixed_pos(self):
        status = [WyckoffPosition(self.pg_num, letter).is_special_fixed() for letter in self.structure_letters]
        return np.all(status)

    def update_pg_number(self, samples=5, verbose=False):
        pg = PlaneGroup(self.pg_num)
        structure_dict = {'C': self.structure_letters}
        atoms = pg.generate_unit_cell_with_sampling(structure_dict, thickness=12., max_samples=samples)
        pg_num_new = get_plane_group(atoms)
        if pg_num_new != self.pg_num:
            if verbose:
                print('plane group number updated from {} to {}.'.format(self.pg_num, pg_num_new))
            self.pg_num = pg_num_new