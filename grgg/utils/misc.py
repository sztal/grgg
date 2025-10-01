from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial, wraps
from inspect import Signature, signature
from itertools import product
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import tree_pformat
from jax.scipy.special import expit

from grgg._typing import IntMatrix, Real, RealMatrix, Reals, RealVector


@jax.jit
def sigmoid(x: Reals) -> Reals:
    """Compute the sigmoid function."""
    return expit(-x)


@jax.jit
def union_probability(*probs: Reals) -> Reals:
    """Compute the union probability of independent events.

    Parameters
    ----------
    p
        Probabilities of individual events.

    Examples
    --------
    >>> union_probability(0.5, 0.5).item()
    0.75
    """
    prob = 1.0
    for p in probs:
        prob *= 1 - p
    return 1 - prob


@jax.jit
def squareform(d: RealVector) -> RealMatrix:
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
    func: Callable[[RealVector, RealVector], Real],
) -> Callable[[RealMatrix, RealMatrix | None], RealVector | RealMatrix]:
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
    >>> from grgg import RandomGenerator
    >>> def euclidean(x, y): return jnp.linalg.norm(x - y)
    >>> pairwise_euclidean = pairwise(euclidean)
    >>> rng = RandomGenerator(0)
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

    def wrapped(
        X: RealMatrix,
        Y: RealMatrix | None = None,
        *,
        condensed: bool = True,
    ) -> RealVector:
        if X.ndim == 1 and (Y is not None and Y.ndim == 1):
            return func(X, Y)
        if X.ndim != 2 or (Y is not None and Y.ndim != 2):
            errmsg = "'X' and 'Y' must be 2D arrays"
            raise ValueError(errmsg)
        if Y is None:

            @jax.jit
            def compute(idx: tuple[int, int]) -> Real:
                return func(X[idx[0]], X[idx[1]])
        else:

            @jax.jit
            def compute(idx: tuple[int, int]) -> Real:
                return func(X[idx[0]], Y[idx[1]])

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


@eqx.filter_jit
def batch_starts(n: int, batch_size: int, *, repeat: int | None = None) -> jnp.ndarray:
    """Get the starting indices of batches.

    Parameters
    ----------
    n
        Total number of items.
    batch_size
        Size of each batch.
    repeat
        If provided, generate all possible starting indices for each batch.

    Returns
    -------
    starts
        An array of starting indices for each batch.

    Examples
    --------
    >>> batch_starts(10, 3)
    Array([0, 3, 6, 9], ...)
    >>> batch_starts(7, 3, repeat=2)
    Array([[0, 0],
           [0, 3],
           [0, 6],
           [3, 0],
           [3, 3],
           [3, 6],
           [6, 0],
           [6, 3],
           [6, 6]], ...)
    """
    if n < 1:
        errmsg = "'n' must be a positive integer."
        raise ValueError(errmsg)
    if batch_size < 1:
        errmsg = "'batch_size' must be a positive integer."
        raise ValueError(errmsg)
    if repeat and repeat < 1:
        errmsg = "'repeat' must be a positive integer."
        raise ValueError(errmsg)
    starts = (jnp.arange(0, n, batch_size),)
    if repeat:
        starts = starts * repeat
    starts = cartesian_product(starts)
    if not repeat or repeat <= 1:
        starts = starts.squeeze()
    return (
        jnp.atleast_1d(starts)
        if repeat is None or repeat <= 1
        else jnp.atleast_2d(starts)
    )


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


def batch_sizes(shape: tuple[int, ...], batch_shape: tuple[int, ...]) -> IntMatrix:
    """Compute batch starts and sizes when using 'batch_shape' on 'shape'."""
    n_batches = tuple(
        (s + bs - 1) // bs for s, bs in zip(shape, batch_shape, strict=True)
    )
    final_sizes = tuple(s // n for s, n in zip(shape, n_batches, strict=True))
    remainders = tuple(s % n for s, n in zip(shape, n_batches, strict=True))
    starts = tuple(
        jnp.arange(0, s, bs) for s, bs in zip(shape, batch_shape, strict=True)
    )
    sizes = tuple(
        jnp.full(len(s), fs) for s, fs in zip(starts, final_sizes, strict=True)
    )
    sizes = tuple(
        s.at[-1].add(r) if r > 0 else s for s, r in zip(sizes, remainders, strict=True)
    )
    starts = jnp.column_stack(starts).squeeze()
    sizes = jnp.column_stack(sizes).squeeze()
    return jnp.stack([starts, sizes], axis=0)


def number_of_batches(n: int, batch_size: int) -> int:
    """Compute the number of batches needed to process `n` with given `batch_size`.

    Parameters
    ----------
    n
        Total number of items.
    batch_size
        Size of each batch.

    Returns
    -------
    n_batches
        The number of batches needed.
    """
    if n <= 0:
        errmsg = "'n' must be a positive integer."
        raise ValueError(errmsg)
    if batch_size <= 0:
        errmsg = "'batch_size' must be a positive integer."
        raise ValueError(errmsg)
    return (n + batch_size - 1) // batch_size


def split_kwargs_by_signature(
    func: Callable, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split keyword arguments by function signature.

    Parameters
    ----------
    signature
        The function signature to match against.

    Returns
    -------
    matched
        A dictionary of matched keyword arguments.
    unmatched
        A dictionary of unmatched keyword arguments.
    """
    sig = func if isinstance(func, Signature) else signature(func)
    matched = {}
    unmatched = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            matched[k] = v
        else:
            unmatched[k] = v
    return matched, unmatched


def format_array(x: jnp.ndarray) -> str:
    """Format a JAX array for display."""
    if jnp.isscalar(x):
        return f"{x.item():.2f}"
    return tree_pformat(x)


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


def is_abstract(cls: type) -> bool:
    """Check if a class is abstract."""
    if bool(getattr(cls, "__abstractmethods__", False)):
        return True
    if bool(getattr(cls, "__abstractvars__", False)):
        return True
    return any(
        getattr(obj, "__isabstractmethod__", False)
        for x in dir(cls)
        if (obj := getattr(cls, x, None))
    )
