from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.ergm.abc import AbstractErgmNodePairView, AbstractErgmNodeView

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

__all__ = ("AbstractRandomGraphNodeView", "AbstractRandomGraphNodePairView")


T = TypeVar("T", bound="AbstractRandomGraph")


class AbstractRandomGraphNodeView[T](AbstractErgmNodeView[T]):
    """Abstract base class for node views of random graph models."""

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


class AbstractRandomGraphNodePairView[T](AbstractErgmNodePairView[T]):
    """Abstract base class for node pair views of random graph models."""

    def probs(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute connection probabilities for selected pairs."""
        return _pairs_probs(self, *args, **kwargs)

    def free_energy(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute edge free energies for selected pairs."""
        return _pairs_free_energy(self, *args, **kwargs)


# Internals --------------------------------------------------------------------------


@eqx.filter_jit
def _pairs_probs(
    pairs: AbstractRandomGraphNodePairView,
    *args: jnp.ndarray,
    **kwargs: Any,
) -> Reals:
    """Compute pairwise connection probabilities."""
    probs = pairs.model.functions.probs(*args, **pairs.parameters, **kwargs)
    if pairs.model.is_homogeneous:
        probs = jnp.full(pairs.shape, probs)
    try:
        i, j = pairs.coords
    except ValueError:
        # This must be a single integer index
        return probs.at[pairs._index].set(0.0)
    return jnp.where(i == j, 0.0, probs)


@eqx.filter_jit
def _pairs_free_energy(
    pairs: AbstractRandomGraphNodePairView,
    *args: jnp.ndarray,
    **kwargs: Any,
) -> Reals:
    """Compute pairwise edge free energies."""
    free_energy = pairs.model.functions.edge_free_energy(
        *args, **pairs.parameters, **kwargs
    )
    if pairs.model.is_homogeneous:
        free_energy = jnp.full(pairs.shape, free_energy)
    try:
        i, j = pairs.coords
    except ValueError:
        # This must be a single integer index
        return free_energy.at[pairs._index].set(jnp.inf)
    return jnp.where(i == j, jnp.nan, free_energy)
