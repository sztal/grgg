from itertools import product

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


def batch_arrays(
    *arrays: np.ndarray,
    batch_size: int = 100,
) -> tuple[np.ndarray, ...]:
    """Yield batches of arrays.

    Parameters
    ----------
    *arrays
        Arrays to be batched. All arrays must have the same first dimension.
    batch_size
        Size of each batch.

    Yields
    ------
    tuple of np.ndarray
        Batches of the input arrays.

    Examples
    --------
    >>> a = np.arange(5)
    >>> b = np.arange(5, 10)
    >>> for batch_a, batch_b in batch_arrays(a, b, batch_size=3):
    ...     print(batch_a, batch_b)
    [0 1 2] [5 6 7]
    [0 1 2] [8 9]
    [3 4] [5 6 7]
    [3 4] [8 9]
    """
    if not arrays:
        return
    if len({len(arr) for arr in arrays}) != 1:
        errmsg = "all input arrays must have the same length"
        raise ValueError(errmsg)
    n = arrays[0].shape[0]
    n_batches = (n + batch_size - 1) // batch_size
    for batch_idx in product(range(n_batches), repeat=len(arrays)):
        batch = []
        for i, arr in zip(batch_idx, arrays, strict=True):
            start = i * batch_size
            end = min(start + batch_size, n)
            batch.append(arr[start:end])
        if batch:
            yield tuple(batch)
