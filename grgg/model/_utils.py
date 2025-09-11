import secrets
from collections.abc import Callable
from functools import singledispatch, wraps

import jax
import jax.numpy as np
from flax import nnx
from jax._src.prng import PRNGKeyArray

from ._typing import Matrix, Scalar, Vector


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
        k: jax.random.key(v if v is not None else secrets.randbelow(0, 2**31))
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

            @nnx.jit
            def compute(idx: tuple[int, int]) -> Scalar:
                return jitted(X[idx[0]], X[idx[1]])
        else:

            @nnx.jit
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

    pairwise_func = nnx.jit(wrapped, static_argnames=("condensed",))
    return wraps(func)(pairwise_func)
