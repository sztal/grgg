from functools import partial, singledispatchmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.models.ergm.abc import AbstractErgmFunctions
from grgg.utils.compute import fori

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

__all__ = ("AbstractRandomGraphFunctions",)


class AbstractRandomGraphFunctions(AbstractErgmFunctions):
    """Abstract base class for random graph model functions."""

    @classmethod
    @eqx.filter_jit
    def couplings(
        cls, params: "AbstractRandomGraph.Parameters", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the couplings of the model."""
        raise NotImplementedError

    @singledispatchmethod
    @classmethod
    @eqx.filter_jit
    def probs(
        cls,
        params: "AbstractRandomGraph.Parameters",
        *args: Any,
        log: bool = False,
        **kwargs: Any,
    ) -> Reals:
        """Compute edge probabilities."""
        couplings = cls.couplings(params, *args, **kwargs)
        return cls.probs(couplings, log=log)

    @probs.register
    @classmethod
    @eqx.filter_jit
    def _(cls, couplings: jnp.ndarray, *, log: bool = False) -> Reals:
        """Compute edge probabilities from couplings."""
        if log:
            return jax.nn.log_sigmoid(-couplings)
        return jax.nn.sigmoid(-couplings)

    @singledispatchmethod
    @classmethod
    @eqx.filter_jit
    def edge_free_energy(
        cls, params: "AbstractRandomGraph.Parameters", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the edge free energy.

        Examples
        --------
        >>> import jax
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> n = 100
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> couplings = model.pairs.couplings()
        >>> efe0 = jax.nn.log_sigmoid(couplings)
        >>> efe1 = model.pairs.free_energy()
        >>> jax.numpy.allclose(efe0, efe1).item()
        True
        """
        couplings = cls.couplings(params, *args, **kwargs)
        return cls.edge_free_energy(couplings)

    @edge_free_energy.register
    @classmethod
    @eqx.filter_jit
    def _(cls, couplings: jnp.ndarray) -> Reals:
        """Compute edge free energy from couplings."""
        return jax.nn.log_sigmoid(couplings)

    @classmethod
    @eqx.filter_jit
    def free_energy(
        cls, model: "AbstractRandomGraph", *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the free energy of the model."""
        vids = jnp.arange(model.n_nodes)

        @jax.jit
        @partial(jax.checkpoint)
        def partial_sum(model: "AbstractRandomGraph", i: Integer) -> Real:
            j = jnp.delete(vids, jnp.array([i]), assume_unique_indices=True)
            return model.pairs[i, j].free_energy(*args, **kwargs).sum()

        @fori(0, model.n_nodes, init=0.0)
        def fe(i: Integer, carry: Real) -> Real:
            return carry + partial_sum(model, i)

        return fe
