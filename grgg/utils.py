import numpy as np


def random_generator(
    random_state: np.random.Generator | int | None = None,
) -> np.random.Generator:
    """Create a :class:`numpy.random.Generator` instance."""
    if not isinstance(random_state, np.random.Generator):
        random_state = np.random.default_rng(random_state)
    if not isinstance(random_state, np.random.Generator):
        errmsg = "'random_state' must be an integer seed or numpy generator."
        raise TypeError(errmsg)
    return random_state
