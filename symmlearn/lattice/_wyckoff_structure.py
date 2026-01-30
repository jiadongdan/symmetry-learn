import numpy as np
from ._wyckoff_position import wyckoff_pos, WyckoffPosition
from ._plane_group import PlaneGroup
from ._layer_group import get_plane_group
from ..utils._ramdom import check_random_state


class WyckoffStructure:

    def __init__(self, pg_num, structure_letters):
        self.structure_letters = list(structure_letters)
        self.pg_num = pg_num
        self.unique_atoms = len(self.structure_letters)

    @property
    def num_atoms(self):
        return sum([len(wyckoff_pos[self.pg_num][letter]) for letter in self.structure_letters])

    def is_all_fixed_pos(self):
        status = [WyckoffPosition(self.pg_num, letter).is_special_fixed() for letter in self.structure_letters]
        return np.all(status)

    def to_structure_dict(self, elements=['C'], seed=None):
        try:
            iter(elements)
        except:
            raise TypeError(f"Elements must be a list or 1d numpy array.")

        n = len(elements)

        if len(self.structure_letters) < n:
            raise ValueError(f"Cannot split {len(self.structure_letters)} items into {n} sections with size >= 1.")

        rng = check_random_state(seed)

        shuffled = np.array(self.structure_letters)
        rng.shuffle(shuffled)

        total = len(self.structure_letters)
        # Reserve 1 item for each section to ensure minimum size of 1
        reserved_count = n
        remaining_count = total - reserved_count

        # Generate proportions for the *remaining* items
        remaining_proportions = rng.dirichlet(np.ones(n))
        raw_additional_sizes = remaining_proportions * remaining_count
        rounded_additional_sizes = np.round(raw_additional_sizes).astype(int)

        # Correct rounding errors for the additional sizes
        adjustment = remaining_count - rounded_additional_sizes.sum()
        rounded_additional_sizes[0] += adjustment

        # Add the reserved 1 item back to each section's size
        sizes = rounded_additional_sizes + 1

        sections = []
        start = 0
        for size in sizes:
            sections.append(shuffled[start:start + size].tolist())
            start += size
        structure_dict = {}
        for element, sect in zip(elements, sections):
            structure_dict[element] = ''.join(sect)
        return structure_dict