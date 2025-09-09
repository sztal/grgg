from collections.abc import Mapping
from itertools import product
from typing import Any

import numpy as np


def parse_switch_flag(
    value: bool | Mapping | None,
    default: bool | None = None,
) -> tuple[bool, Mapping[str, Any]]:
    """Get option `value`.

    Parameters
    ----------
    value
        Option value.
    default
        Default value to fill in when `value` is `None`.

    Returns
    -------
    value
        The resulting value.
    options
        A empty dict or `value` if it was passed as a mapping.

    Examples
    --------
    >>> parse_switch_flag(None)
    (False, {})
    >>> parse_switch_flag(None, default=True)
    (True, {})
    >>> parse_switch_flag(True)
    (True, {})
    >>> parse_switch_flag(False)
    (False, {})
    >>> parse_switch_flag({"a": 1, "b": 2})
    (True, {'a': 1, 'b': 2})
    >>> parse_switch_flag({}, default=True)
    (True, {})
    >>> parse_switch_flag({})
    (True, {})
    """
    if value is None:
        value = default
    if isinstance(value, Mapping):
        value, options = True, value
    else:
        value, options = bool(value), {}
    return value, options


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


def make_grid(n: int, k: int, ranges: list[tuple[float, float]]) -> np.ndarray:
    """Create a regular grid of `n` points in `k` dimensions.

    Parameters
    ----------
    n : int
        Number of points. The total number of points may be a bit less than `n`,
        since the number of points per dimension is rounded down to the nearest integer.
    k : int
        Number of dimensions.
    ranges : list of tuples
        List of (min, max) tuples for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (n**k, k), where each row is a point coordinate.

    Examples
    --------
    >>> make_grid(9, 2, [(-1, 1), (0, 2)])
    array([[-1.,  0.],
           [-1.,  1.],
           [-1.,  2.],
           [ 0.,  0.],
           [ 0.,  1.],
           [ 0.,  2.],
           [ 1.,  0.],
           [ 1.,  1.],
           [ 1.,  2.]])
    """
    n = max(1, int(np.ceil(n ** (1 / k))))
    linspaces = [np.linspace(start, stop, n) for (start, stop) in ranges]
    mesh = np.meshgrid(*linspaces, indexing="ij")
    grid_points = np.stack(mesh, axis=-1).reshape(-1, k)
    return grid_points
