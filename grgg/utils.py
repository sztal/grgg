import secrets
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import singledispatch, wraps
from itertools import product
from typing import Any

import jax
import jax.numpy as np
from flax import nnx
from jax._src.prng import PRNGKeyArray

from grgg._typing import Matrix, Scalar, Vector


def random_keys(
    default: int | None = None,
    **seeds: int | None,
) -> dict[str, PRNGKeyArray]:
    """Generate a random key for JAX operations.

    Parameters
    ----------
    default
        If provided, use this integer as the seed for the random key.
        Otherwise, generate a random seed using :mod:`secrets`.
    **streams
        Additional named streams with their own integer seeds,
        which can also be `None` to generate a random seed.
    """
    seeds = {"default": default, **seeds}
    keys = {
        k: jax.random.key(v if v is not None else secrets.randbelow(2**31))
        for k, v in seeds.items()
    }
    return keys


@singledispatch
def random_state(
    default: int | None | nnx.RngStream,
    **seeds: int | None | nnx.RngStream,
) -> nnx.Rngs:
    """Create a random state for Flax modules.

    Arguments are passed to :func:`random_keys`.
    """
    keys = random_keys(default, **seeds)
    return nnx.Rngs(**keys)


@random_state.register(nnx.Rngs)
def _(rngs: nnx.Rngs, **seeds: int | None | nnx.RngStream) -> nnx.Rngs:
    if not seeds:
        return rngs
    streams = {k: v for k, v in rngs.__dict__.items() if isinstance(v, nnx.RngStream)}
    streams = {**streams, **seeds}
    return nnx.Rngs(**streams)


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
    D = np.zeros_like(d, shape=(n, n))
    D = D.at[np.triu_indices_from(D, k=1)].set(d)
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
    >>> def euclidean(x, y): return np.linalg.norm(x - y)
    >>> pairwise_euclidean = pairwise(euclidean)
    >>> rngs = nnx.Rngs(0)
    >>> X = rngs.normal((100, 3))
    >>> Y = rngs.normal((20, 3))
    >>> D1 = pairwise_euclidean(X)
    >>> C1 = pairwise_euclidean(X, Y)  # cross-distance

    Test the pairwise case against the reference implementation from :mod:`scipy`.
    >>> from scipy.spatial.distance import pdist, squareform
    >>> D2 = pdist(X)
    >>> bool(np.allclose(D1, D2))
    True
    >>> D1_square = pairwise_euclidean(X, condensed=False)
    >>> D2_square = squareform(D2)
    >>> bool(np.allclose(D1_square, D2_square))
    True

    Test the cross-distance case against the reference implementation from :mod:`scipy`.
    >>> from scipy.spatial.distance import cdist
    >>> C2 = cdist(X, Y)
    >>> bool(np.allclose(C1, C2))
    True

    Importantly, the compiled function still works on single pairs of vectors.
    >>> x = np.array([1])
    >>> y = np.array([2])
    >>> float(pairwise_euclidean(x, y))
    1.0
    """
    jitted = nnx.jit(func)

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
            indices = np.array(np.triu_indices(len(X), k=1)).T
        else:
            grid = np.meshgrid(np.arange(len(X)), np.arange(len(Y)), indexing="ij")
            indices = np.stack([x.ravel() for x in grid], axis=1)
        output = jax.vmap(compute)(indices)
        if Y is None and not condensed:
            output = squareform(output)
        elif Y is not None:
            output = output.reshape(len(X), len(Y))
        return output

    pairwise_func = jax.jit(wrapped, static_argnames=("condensed",))
    return wraps(func)(pairwise_func)


def cartesian_product(
    arrays: Sequence[np.ndarray], axis: int | None = None
) -> np.ndarray:
    """Compute the cartesian product of input arrays along a given axis.

    Parameters
    ----------
    arrays
        Sequence of arrays to compute the cartesian product of.
        The arrays must have the same shape except along the specified axis.
    axis
        The axis along which the cartesian product is computed.
        This is carried out using :func:`np.stack` and it is expected
        that the concatenation axis exists in all input arrays.
        If `None`, then :func:`np.column_stack` is used instead,
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
    >>> a = np.array([1, 2])
    >>> b = np.array([3, 4])
    >>> cartesian_product([a, b])
    Array([[1, 3],
           [1, 4],
           [2, 3],
           [2, 4]], ...)

    >>> c = np.array([5, 6])
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
    >>> a = np.arange(27).reshape(3, 3, 3)
    >>> b = np.arange(18).reshape(3, 2, 3)
    >>> cartesian_product([a, b], axis=1).shape
    (3, 2, 6, 3)
    """
    arrays = [np.asarray(a) for a in arrays]
    if not arrays:
        return np.array([]).reshape(0, 0)
    if len(arrays) == 1:
        return np.expand_dims(arrays[0], axis=axis or -1)
    if len(arrays) == 2:
        a, b = arrays
        ai = np.arange(a.shape[axis or 0])
        bi = np.arange(b.shape[axis or 0])
        grid = np.stack(np.meshgrid(ai, bi, indexing="ij"), axis=-1).reshape(-1, 2)
        a = np.take(a, grid[:, 0], axis=axis or 0)
        b = np.take(b, grid[:, 1], axis=axis or 0)
        return np.column_stack([a, b]) if axis is None else np.stack([a, b], axis=axis)
    product = cartesian_product(arrays[:2], axis=axis)
    return cartesian_product([product, *arrays[2:]], axis=axis)


cartesian_product = jax.jit(cartesian_product, static_argnames=("axis",))


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
