from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from itertools import product
from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Matrix, Scalar, Vector


@jax.jit
def squareform(d: Vector) -> Matrix:
    """Convert a condensed distance matrix to a square form.

    Condensed vector always refers to the upper triangle of the square matrix.

    Parameters
    ----------
    X
        Condensed distance matrix, shape (m * (m - 1) / 2,).

    Returns
    -------
    D
        Square distance matrix, shape (m, m).
    """
    n = int((1 + (1 + 8 * len(d)) ** 0.5) / 2)
    D = jnp.zeros_like(d, shape=(n, n))
    D = D.at[jnp.triu_indices_from(D, k=1)].set(d)
    return D + D.T


def pairwise(
    func: Callable[[Vector, Vector], Scalar],
) -> Callable[[Matrix, Matrix | None], Vector | Matrix]:
    """Convert a function on two vectors to one returning pairwise distances.

    The resulting function will take a matrix as the only required argument,
    and compute the pairwise distances between its rows using the provided function.
    If the second matrix is not provided, it computes the pairwise distances between
    all rows of the first matrix and themselves. It also accepts a keyword argument
    `condensed` (default `True`) to return either a condensed distance matrix
    or a full square distance matrix in the case of a single input matrix. Otherwise,
    it is ignored. Additional arguments cannot be passed after compilation.

    Condensed vector always refers to the upper triangle of the square matrix.

    Examples
    --------
    >>> from grgg.random import RandomGenerator
    >>> def euclidean(x, y): return jnp.linalg.norm(x - y)
    >>> pairwise_euclidean = pairwise(euclidean)
    >>> rng = RandomGenerator.from_seed(0)
    >>> X = rng.normal((100, 3))
    >>> Y = rng.normal((20, 3))
    >>> D1 = pairwise_euclidean(X)
    >>> C1 = pairwise_euclidean(X, Y)  # cross-distance

    Test the pairwise case against the reference implementation from :mod:`scipy`.
    >>> from scipy.spatial.distance import pdist, squareform
    >>> D2 = pdist(X)
    >>> bool(jnp.allclose(D1, D2))
    True
    >>> D1_square = pairwise_euclidean(X, condensed=False)
    >>> D2_square = squareform(D2)
    >>> bool(jnp.allclose(D1_square, D2_square))
    True

    Test the cross-distance case against the reference implementation from :mod:`scipy`.
    >>> from scipy.spatial.distance import cdist
    >>> C2 = cdist(X, Y)
    >>> bool(jnp.allclose(C1, C2))
    True

    Importantly, the compiled function still works on single pairs of vectors.
    >>> x = jnp.array([1])
    >>> y = jnp.array([2])
    >>> float(pairwise_euclidean(x, y))
    1.0
    """
    jitted = jax.jit(func)

    def wrapped(
        X: Matrix,
        Y: Matrix | None = None,
        *,
        condensed: bool = True,
    ) -> Vector:
        if X.ndim == 1 and (Y is not None and Y.ndim == 1):
            return jitted(X, Y)
        if X.ndim != 2 or (Y is not None and Y.ndim != 2):
            errmsg = "'X' and 'Y' must be 2D arrays"
            raise ValueError(errmsg)
        if Y is None:

            @jax.jit
            def compute(idx: tuple[int, int]) -> Scalar:
                return jitted(X[idx[0]], X[idx[1]])
        else:

            @jax.jit
            def compute(idx: tuple[int, int]) -> Scalar:
                return jitted(X[idx[0]], Y[idx[1]])

        if Y is None:
            indices = jnp.array(jnp.triu_indices(len(X), k=1)).T
        else:
            grid = jnp.meshgrid(jnp.arange(len(X)), jnp.arange(len(Y)), indexing="ij")
            indices = jnp.stack([x.ravel() for x in grid], axis=1)
        output = jax.vmap(compute)(indices)
        if Y is None and not condensed:
            output = squareform(output)
        elif Y is not None:
            output = output.reshape(len(X), len(Y))
        return output

    pairwise_func = jax.jit(wrapped, static_argnames=("condensed",))
    return wraps(func)(pairwise_func)


@partial(jax.jit, static_argnames=("axis",))
def cartesian_product(
    arrays: Sequence[jnp.ndarray], axis: int | None = None
) -> jnp.ndarray:
    """Compute the cartesian product of input arrays along a given axis.

    Parameters
    ----------
    arrays
        Sequence of arrays to compute the cartesian product of.
        The arrays must have the same shape except along the specified axis.
    axis
        The axis along which the cartesian product is computed.
        This is carried out using :func:`jnp.stack` and it is expected
        that the concatenation axis exists in all input arrays.
        If `None`, then :func:`jnp.column_stack` is used instead,
        which requires all input arrays to be of the same shape
        and concatenates them along a new last axis.

    Returns
    -------
    product
        The cartesian product of the input arrays.
        For two input arrays the output has `ndim+1` dimensions.
        In no case the output will have less than 2 dimensions.

    Examples
    --------
    >>> a = jnp.array([1, 2])
    >>> b = jnp.array([3, 4])
    >>> cartesian_product([a, b])
    Array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]], ...)

    >>> c = jnp.array([5, 6])
    >>> cartesian_product([a, b, c])
    Array([[1, 3, 5],
           [1, 3, 6],
           [1, 4, 5],
           [1, 4, 6],
           [2, 3, 5],
           [2, 3, 6],
           [2, 4, 5],
           [2, 4, 6]], ...)

    Product along arbitrary axis are also possible.
    >>> a = jnp.arange(27).reshape(3, 3, 3)
    >>> b = jnp.arange(18).reshape(3, 2, 3)
    >>> cartesian_product([a, b], axis=1).shape
    (3, 2, 6, 3)
    """
    arrays = [jnp.asarray(a) for a in arrays]
    if not arrays:
        return jnp.array([]).reshape(0, 0)
    if len(arrays) == 1:
        return jnp.expand_dims(arrays[0], axis=axis or -1)
    if len(arrays) == 2:
        a, b = arrays
        ai = jnp.arange(a.shape[axis or 0])
        bi = jnp.arange(b.shape[axis or 0])
        grid = jnp.stack(jnp.meshgrid(ai, bi, indexing="ij"), axis=-1).reshape(-1, 2)
        a = jnp.take(a, grid[:, 0], axis=axis or 0)
        b = jnp.take(b, grid[:, 1], axis=axis or 0)
        return (
            jnp.column_stack([a, b]) if axis is None else jnp.stack([a, b], axis=axis)
        )
    product = cartesian_product(arrays[:2], axis=axis)
    return cartesian_product([product, *arrays[2:]], axis=axis)


def split_by(
    data: jnp.ndarray, groups: Sequence | jnp.ndarray
) -> tuple[jnp.ndarray, ...]:
    """Split an array into groups along the first axis.

    Parameters
    ----------
    data
        The array to be split.
    groups
        A sequence of group labels.

    Returns
    -------
    splits
        A tuple of arrays, each corresponding to a group.

    Examples
    --------
    >>> data = jnp.arange(10).reshape(5, 2)
    >>> data
    Array([[0, 1],
           [2, 3],
           [4, 5],
           [6, 7],
           [8, 9]], ...)
    >>> split_by(data, [1, 1, 2, 2, 1])
    (Array([[0, 1],
           [2, 3],
           [8, 9]], dtype=int32), Array([[4, 5],
           [6, 7]], dtype=int32))
    """
    groups = jnp.atleast_1d(jnp.asarray(groups))
    if groups.ndim != 1:
        errmsg = "'groups' must be a 1D array-like."
        raise ValueError(errmsg)
    if len(groups) != len(data):
        errmsg = "'groups' must have the same length as the first dimension of 'data'."
        raise ValueError(errmsg)
    order = jnp.argsort(groups, stable=True)
    _, splitpoints = jnp.unique(groups[order], return_index=True)
    return tuple(jnp.split(data[order], splitpoints[1:]))


def batch_slices(
    n: int,
    batch_size: int,
    *,
    repeat: int | None = None,
) -> Iterator[slice] | Iterator[tuple[slice, ...]]:
    """Iterate over batch slices.

    Parameters
    ----------
    n
        Total number of items.
    batch_size
        Size of each batch.
    repeat
        If provided, generate all possible pairs of batch slices,
        or higher cartesian products if repeat > 2.

    Yields
    ------
    slice or tuple of slices
        Slice object for the current batch if `repeat == None`.
        Otherwise a tuple of slice objects for each batch.
    """

    def _iter():
        n_batches = (n + batch_size - 1) // batch_size
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n)
            yield slice(start, end)

    if not repeat:
        yield from _iter()
    else:
        yield from product(_iter(), repeat=repeat)


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
