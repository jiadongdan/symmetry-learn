from itertools import combinations

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
