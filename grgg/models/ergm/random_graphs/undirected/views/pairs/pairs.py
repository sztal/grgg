from typing import TYPE_CHECKING, ClassVar, TypeVar

from grgg.models.ergm.random_graphs.abc import (
    AbstractRandomGraphNodePairView,
)
from grgg.models.ergm.random_graphs.undirected.motifs.pairs.motifs import (
    RandomGraphNodePairMotifs,
)

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import RandomGraph

__all__ = ("RandomGraphNodePairView",)


T = TypeVar("T", bound="RandomGraph")
ME = TypeVar("ME", bound=RandomGraphNodePairMotifs)


class RandomGraphNodePairView[T, ME](AbstractRandomGraphNodePairView[T, ME]):
    r"""Node pair view for undirected random graph models.

    Examples
    --------
    Define a homogeneous undirected random graph model which is equivalent to the
    `(n, p)`-Erdős–Rényi model with :math:`p = 1 / (1 + e^{-\mu})`.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph
    >>> model = RandomGraph(100, mu=-2)

    By default, homogeneous models return a scalar value for connection probabilities.
    >>> model.pairs.probs().item()
    0.11920293

    However, a full shape (or any arbitrary indexing) can be requested by passing
    indices to the pairs view.
    >>> probs = model.pairs[...].probs()
    >>> probs.shape
    (100, 100)

    Diagonal entries are zeros by definitions since graphs do not contain self-loops.
    >>> jnp.all(probs.diagonal() == 0).item()
    True

    And off diagonal entries are equal to the connection probability.
    >>> jnp.all(probs[~jnp.eye(100, dtype=bool)] == model.pairs.probs()).item()
    True

    The convention is kept even for arbitrary indexing.
    >>> probs = model.pairs[2:5, 3:7].probs()
    >>> probs.shape
    (3, 4)
    >>> i = jnp.array([2, 3, 4])
    >>> j = jnp.array([3, 4, 5, 6])
    >>> I, J = jnp.ix_(i, j)
    >>> probs = model.pairs[I, J].probs()
    >>> jnp.all(probs[I != J] == model.pairs.probs()).item()
    True
    >>> jnp.all(probs[I == J] == 0).item()
    True

    Now we do the same for a heterogeneous model.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> mu = rng.normal(100)
    >>> model = RandomGraph(mu.size, mu=mu)
    >>> probs = model.pairs.probs()
    >>> probs.shape
    (100, 100)
    >>> jnp.all(probs.diagonal() == 0).item()
    True
    >>> jnp.all(probs.T == probs).item()
    True
    >>> jnp.all(probs[~jnp.eye(100, dtype=bool)] != 0).item()
    True
    >>> i = jnp.array([2, 3, 4])
    >>> j = jnp.array([3, 4, 5, 6])
    >>> I, J = jnp.ix_(i, j)
    >>> probs = model.pairs[I, J].probs()
    >>> jnp.all(probs[I != J] != 0).item()
    True
    >>> jnp.all(probs[I == J] == 0).item()
    True
    """

    motifs_cls: ClassVar[type[RandomGraphNodePairMotifs]] = RandomGraphNodePairMotifs
