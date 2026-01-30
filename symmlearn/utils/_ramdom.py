import numpy as np

def check_random_state(seed):
    """Convert seed to np.random.Generator.

    Parameters
    ----------
    seed : None, int, RandomState, or Generator

    Returns
    -------
    np.random.Generator
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, int):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.RandomState):
        # Use RandomState to generate a seed for Generator
        return np.random.default_rng(seed.randint(2**31))
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(f"Invalid seed type: {type(seed)}")