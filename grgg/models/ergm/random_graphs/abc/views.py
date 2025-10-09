from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.ergm.abc import AbstractErgmNodePairView, AbstractErgmNodeView

if TYPE_CHECKING:
    from .models import AbstractRandomGraph

__all__ = ("AbstractRandomGraphNodeView", "AbstractRandomGraphNodePairView")


T = TypeVar("T", bound="AbstractRandomGraph")


class AbstractRandomGraphNodeView[T, V](AbstractErgmNodeView[T, V]):
    """Abstract base class for node views of random graph models."""

    model: eqx.AbstractVar[T]

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize a new model with the current selection of nodes.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = UndirectedRandomGraph(mu.size, mu)
        >>> nodes = model.nodes[:10]
        >>> nodes.n_nodes
        10
        >>> nodes.model.n_nodes
        100
        >>> submodel = nodes.materialize()
        >>> submodel.n_nodes
        10
        >>> submodel.parameters.mu.shape
        (10,)
        >>> jnp.all(submodel.pairs.probs() == model.pairs[:10, :10].probs()).item()
        True
        """
        return super().materialize(copy=copy)


class AbstractRandomGraphNodePairView[T, E](AbstractErgmNodePairView[T, E]):
    """Abstract base class for node pair views of random graph models."""

    model: eqx.AbstractVar[T]

    def probs(self, *args: Any, adjust_quantized: bool = False, **kwargs: Any) -> Reals:
        """Compute connection probabilities for selected pairs.

        Parameters
        ----------
        adjust_quantized
            boolean flag to indicate whether to adjust for self-loops when
            computing connection probabilities between bins in quantized models.
        """
        return _pairs_probs(self, *args, adjust_quantized=adjust_quantized, **kwargs)


# Internals --------------------------------------------------------------------------


@eqx.filter_jit
def _pairs_probs(
    pairs: AbstractRandomGraphNodePairView,
    *args: jnp.ndarray,
    adjust_quantized: bool = False,
) -> Reals:
    """Compute pairwise connection probabilities."""
    probs = pairs.model.probability(*args, pairs.model.coupling(*pairs.parameters))
    if pairs.model.is_quantized and not adjust_quantized:
        return probs
    if pairs.model.is_homogeneous:
        probs = jnp.full(pairs.shape, probs)
    try:
        i, j = pairs.coords
    except ValueError:
        # This must be a single integer index
        if adjust_quantized and pairs.model.is_quantized:
            w = pairs.model.quantizer.weights
            return probs.at[pairs._index].mul(1 / w * (w - 1))
        return probs.at[pairs._index].set(0.0)
    if adjust_quantized and pairs.model.is_quantized:
        w = pairs.model.quantizer.weights[i]
        return jnp.where(i == j, probs / w * (w - 1), probs)
    return jnp.where(i == j, 0.0, probs)
