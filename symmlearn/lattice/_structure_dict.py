import numpy as np
from typing import Iterable, List, Tuple, Optional
from itertools import combinations, combinations_with_replacement
from ._wyckoff_position import wyckoff_pos, WyckoffPosition
from ._wyckoff_structure import WyckoffStructure
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

def all_multisets(
        elements: Iterable,
        max_size: Optional[int] = None,
        exclude_empty: bool = True
) -> List[Tuple]:
    """
    Generate all multisets from elements up to max_size.

    Parameters
    ----------
    elements : Iterable
        The base set of elements to generate multisets from.
    max_size : int, optional
        Maximum size of multisets to generate. If None, defaults to len(elements).
    exclude_empty : bool, default=True
        Whether to exclude the empty multiset from results.

    Returns
    -------
    List[Tuple]
        List of all multisets represented as tuples, sorted by size then lexicographically.

    Examples
    --------
    >>> all_multisets(['a', 'b'], max_size=2)
    [('a',), ('b',), ('a', 'a'), ('a', 'b'), ('b', 'b')]

    >>> all_multisets(['a', 'b'], max_size=2, exclude_empty=False)
    [(), ('a',), ('b',), ('a', 'a'), ('a', 'b'), ('b', 'b')]
    """
    # Convert to list to allow len() and ensure consistent ordering
    elements = list(elements)

    if not elements:
        return [] if exclude_empty else [()]

    if max_size is None:
        max_size = len(elements)

    if max_size < 0:
        raise ValueError(f"max_size must be non-negative, got {max_size}")

    # Start with empty multiset if not excluded
    result = [] if exclude_empty else [()]

    # Generate multisets of each size
    for size in range(1, max_size + 1):
        result.extend(combinations_with_replacement(elements, size))

    return result

def all_power_sets(
        elements: Iterable,
        max_size: Optional[int] = None,
        min_size: int = 0,
        exclude_empty: bool = False
) -> List[Tuple]:
    """
    Generate the power set (all subsets) of elements.

    Parameters
    ----------
    elements : Iterable
        The base set of elements to generate subsets from.
    max_size : int, optional
        Maximum size of subsets to generate. If None, includes all sizes up to len(elements).
    min_size : int, default=0
        Minimum size of subsets to generate.
    exclude_empty : bool, default=False
        Whether to exclude the empty set from results.

    Returns
    -------
    List[Tuple]
        List of all subsets represented as tuples, sorted by size then lexicographically.

    Examples
    --------
    >>> all_power_set(['a', 'b'])
    [(), ('a',), ('b',), ('a', 'b')]

    >>> all_power_set(['a', 'b', 'c'], max_size=2)
    [(), ('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c'), ('b', 'c')]

    >>> all_power_set(['a', 'b'], exclude_empty=True)
    [('a',), ('b',), ('a', 'b')]

    >>> all_power_set(['a', 'b', 'c'], min_size=2, max_size=2)
    [('a', 'b'), ('a', 'c'), ('b', 'c')]
    """
    # Convert to list to allow len() and ensure consistent ordering
    elements = list(elements)

    if not elements:
        return [] if exclude_empty else [()]

    n = len(elements)

    if max_size is None:
        max_size = n

    # Validate inputs
    if min_size < 0:
        raise ValueError(f"min_size must be non-negative, got {min_size}")
    if max_size < 0:
        raise ValueError(f"max_size must be non-negative, got {max_size}")
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) cannot be greater than max_size ({max_size})")

    # Adjust for exclude_empty
    effective_min_size = max(min_size, 1 if exclude_empty else 0)

    # Generate subsets of each size
    result = []
    for size in range(effective_min_size, min(max_size + 1, n + 1)):
        result.extend(combinations(elements, size))

    return result

def remove_duplicate_pairs(pairs):
    """
    From a list of 2-element pairs [A, B], drop duplicates such that
    [A, B] and [B, A] count as the same.
    """
    seen = set()
    unique = []
    for a, b in pairs:
        # make a key that doesn't care about order
        key = tuple(sorted((tuple(a), tuple(b))))
        if key not in seen:
            seen.add(key)
            unique.append([a, b])
    return unique

def remove_groups_with_duplicate_fixed_positions(pg_num: int,
                                                 groups: list[list[list[str]]]
                                                 ) -> list[list[list[str]]]:
    """
    From a list of letter‐groups (each a list of sub‐lists of Wyckoff letters),
    drop any group where the same fully numeric (i.e. “fixed”) coordinate
    appears more than once. Coordinates involving ‘x’ or ‘y’ are treated as
    variable and may repeat without issue.

    Args:
        pg_num: int
            Plane‐group number (a key in wyckoff_pos).
        groups: List[List[List[str]]]
            Each entry is a list of lists of Wyckoff letters,
            e.g. [['a'], ['a','b'], ['a']] or longer.

    Returns:
        List[List[List[str]]]
            Filtered groups with no duplicate fixed coords.
    """
    filtered = []
    for group in groups:
        # 1) Gather all coordinate expressions for this group
        coords = []
        for letter_list in group:
            for letter in letter_list:
                coords.extend(wyckoff_pos[pg_num][letter])

        # 2) Strip spaces
        normed = [c.replace(' ', '') for c in coords]

        # 3) Keep only fully numeric/fractional coords
        fixed = [c for c in normed if ('x' not in c and 'y' not in c)]

        # 4) If none repeat, keep it
        if len(fixed) == len(set(fixed)):
            filtered.append(group)

    return filtered

def get_counts(list_of_letters):
    return sum([len(wyckoff_pos[pg_num][letter]) for letters in list_of_letters for letter in letters])

def get_all_A_types(pg_num = 6, max_counts=12):
    letters = list(wyckoff_pos[pg_num].keys())
    ps = power_set(letters)
    ps = np.array(ps, dtype=object)
    counts = np.array([sum([len(wyckoff_pos[pg_num][k]) for k in ks]) for ks in ps])
    inds = np.where(counts <=max_counts)[0]
    aa = [ps[ind] for ind in inds]
    return aa



def get_all_AB_structure(pg_num = 6, max_counts=8):
    # get all wyckoff letters from plane group number
    letters = list(wyckoff_pos[pg_num].keys())
    ps = power_set(letters)
    # get the pairs
    aa = [[i, j] for i in ps for j in ps]

    counts_aa = np.array([get_counts(e) for e in aa])
    inds = np.where(counts_aa <=max_counts)[0]
    aa = [aa[ind] for ind in inds]

    bb = remove_duplicate_pairs(aa)
    cc = remove_groups_with_duplicate_fixed_positions(pg_num, bb)

    return [{'C': e1, 'B':e2} for (e1, e2) in cc]


def split_wyckoff_letters(pg_num):
    letters_fixed_pos = []
    letters_variable_pos = []
    letters = list(wyckoff_pos[pg_num].keys())
    for letter in letters:
        if WyckoffPosition(pg_num,letter).is_special_fixed():
            letters_fixed_pos.append(letter)
        else:
            letters_variable_pos.append(letter)
    return np.array(letters_fixed_pos), np.array(letters_variable_pos)


def get_structure_letters(pg_num, max_counts=8):
    fixed_letters, variable_letters = split_wyckoff_letters(pg_num=pg_num)
    fixed_set = all_power_sets(fixed_letters, exclude_empty=False)

    variable_multiplicity = np.array([WyckoffPosition(pg_num, letter).multiplicity for letter in variable_letters])
    max_size = max_counts // variable_multiplicity.min()
    variable_set = all_multisets(variable_letters, max_size=max_size, exclude_empty=False)

    variable_set = np.array(variable_set, dtype=object)

    fixed_set_counts = np.array([WyckoffStructure(pg_num, letters).num_atoms for letters in fixed_set])
    variable_set_counts = np.array([WyckoffStructure(pg_num, letters).num_atoms for letters in variable_set])

    final_list = []
    for i, e1 in enumerate(fixed_set):
        # e1 is a list
        counts = max_counts - fixed_set_counts[i]
        mask = variable_set_counts <= counts
        for e2 in variable_set[mask]:
            # e1 is also a list
            final_list.append(''.join(e1+e2))
    return final_list[1:]

def mix_combination(pg_num,max_counts=8):
    fixed_letters, general_letters = split_wyckoff_letters(pg_num=pg_num)
    fixed_counts = np.array([WyckoffPosition(pg_number=pg_num,letter=letter).multiplicity for letter in fixed_letters])
    general_counts = np.array([WyckoffPosition(pg_number=pg_num,letter=letter).multiplicity for letter in general_letters])
    fixed_combs = []
    fixed_combs_counts = []
    for i in range(len(fixed_letters) + 1):
        fixed_letter_comb = list(combinations(fixed_letters, i))
        for c in fixed_letter_comb:
            fixed_combs.append(c)
        fixed_count = list(combinations(fixed_counts, i))
        for cc in fixed_count:
            fixed_combs_counts.append(cc)
    fixed_combs_counts = np.array([np.sum(i) for i in fixed_combs_counts])
    fixed_combs = np.array(fixed_combs, dtype=object)

    general_combs = []
    general_combs_counts = []
    min_length = np.min(general_counts)
    for i in range(max_counts//min_length+1):
        general_letter_comb = list(combinations_with_replacement(general_letters, i))
        for c in general_letter_comb:
            general_combs.append(c)
        general_count = list(combinations_with_replacement(general_counts, i))
        for cc in general_count:
            general_combs_counts.append(cc)
    general_combs_counts = np.array([np.sum(i) for i in general_combs_counts])
    general_combs = np.array(general_combs, dtype=object)

    final_comb = []
    for count,f_comb in zip(fixed_combs_counts, fixed_combs):
        general_max_count = max_counts-count
        mask = general_combs_counts <= general_max_count
        g_combs = general_combs[mask]
        for g_comb in g_combs:
            if len(f_comb) > 0:
                final_comb.append(f_comb+g_comb)
            else:
                final_comb.append(g_comb)
    return final_comb