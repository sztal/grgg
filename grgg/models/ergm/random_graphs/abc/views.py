from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.ergm.abc import AbstractErgmNodePairView, AbstractErgmNodeView

if TYPE_CHECKING:
    from .models import AbstractRandomGraph
    from .motifs import AbstractRandomGraphNodeMotifs, AbstractRandomGraphNodePairMotifs

    MV = TypeVar("MV", bound=AbstractRandomGraphNodeMotifs)
    ME = TypeVar("ME", bound=AbstractRandomGraphNodePairMotifs)

__all__ = ("AbstractRandomGraphNodeView", "AbstractRandomGraphNodePairView")


T = TypeVar("T", bound="AbstractRandomGraph")


class AbstractRandomGraphNodeView[T, MV](AbstractErgmNodeView[T, MV]):
    """Abstract base class for node views of random graph models."""

    model: eqx.AbstractVar[T]

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize a new model with the current selection of nodes.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = RandomGraph(mu.size, mu)
        >>> nodes = model.nodes[:10]
        >>> nodes.n_nodes
        10
        >>> nodes.model.n_nodes
        100
        >>> submodel = nodes.materialize()
        >>> submodel.n_nodes
        10
        >>> submodel.mu.shape
        (10,)
        >>> jnp.all(submodel.pairs.probs() == model.pairs[:10, :10].probs()).item()
        True
        """
        return super().materialize(copy=copy)


class AbstractRandomGraphNodePairView[T, ME](AbstractErgmNodePairView[T, ME]):
    """Abstract base class for node pair views of random graph models."""

    model: eqx.AbstractVar[T]

    def probs(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute connection probabilities for selected pairs."""
        return _pairs_probs(self, *args, **kwargs)


# Internals --------------------------------------------------------------------------


@eqx.filter_jit
def _pairs_probs(
    pairs: AbstractRandomGraphNodePairView,
    *args: jnp.ndarray,
) -> Reals:
    """Compute pairwise connection probabilities."""
    probs = pairs.model.probability(*args, pairs.model.coupling(*pairs.parameters))
    if pairs.model.is_homogeneous:
        probs = jnp.full(pairs.shape, probs)
    try:
        i, j = pairs.coords
    except ValueError:
        # This must be a single integer index
        return probs.at[pairs._index].set(0.0)
    return jnp.where(i == j, 0.0, probs)
