from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.models.ergm.abc import AbstractErgmFunctions
from grgg.utils.compute import fori

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

    T = TypeVar("T", bound=AbstractRandomGraph)

__all__ = ("AbstractRandomGraphFunctions",)


class AbstractRandomGraphFunctions[T](AbstractErgmFunctions[T]):
    """Abstract base class for random graph model functions."""

    def probs(self, params: "T.Parameters", **kwargs: Any) -> Reals:
        """Compute edge probabilities."""
        return probs(self, params, **kwargs)

    def edge_free_energy(self, params: "T.Parameters", **kwargs: Any) -> Reals:
        """Compute the edge free energy."""
        return edge_free_energy(self, params, **kwargs)

    def free_energy(self, *args: Any, **kwargs: Any) -> Real:
        """Compute the free energy of the model."""
        return free_energy(self, *args, **kwargs)


@eqx.filter_jit
def probs(
    funcs: AbstractRandomGraphFunctions,
    params: "T.Parameters",
    *,
    log: bool = False,
    **kwargs: Any,
) -> Reals:
    """Compute edge probabilities."""
    couplings = funcs.couplings(params, **kwargs)
    if log:
        return jax.nn.log_sigmoid(-couplings)
    return jax.nn.sigmoid(-couplings)


@eqx.filter_jit
def edge_free_energy(
    funcs: AbstractRandomGraphFunctions, params: "T.Parameters", **kwargs: Any
) -> Reals:
    """Compute the edge free energy."""
    couplings = funcs.couplings(params, **kwargs)
    return -jax.nn.log_sigmoid(couplings)


@eqx.filter_jit
def free_energy(
    funcs: AbstractRandomGraphFunctions, *args: Any, **kwargs: Any
) -> Reals:
    """Compute the free energy of the model."""
    vids = jnp.arange(funcs.model.n_nodes)

    @jax.jit
    @partial(jax.checkpoint)
    def partial_sum(model: "AbstractRandomGraph", i: Integer) -> Real:
        j = jnp.delete(vids, jnp.array([i]), assume_unique_indices=True)
        return model.pairs[i, j].free_energy(*args, **kwargs).sum()

    @fori(1, funcs.model.n_nodes, init=0.0)
    def fe(i: Integer, carry: Real) -> Real:
        return carry + partial_sum(funcs.model, i)

    return fe
