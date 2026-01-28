import numpy as np
from typing import Iterable, List, Tuple, Optional
from itertools import combinations, combinations_with_replacement
from ._wyckoff_position import wyckoff_pos, WyckoffPosition
from ._wyckoff_structure import WyckoffStructure


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
            # concatenate e1 and e2, we get a single list of letters, then use .join to make it a single string
            final_list.append(''.join(e1+e2))
    # remove the first one, which is an empty
    return final_list[1:]

