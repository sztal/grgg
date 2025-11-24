from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Real, Reals
from grgg.models.abc import AbstractModelFunctions

if TYPE_CHECKING:
    from grgg.models.ergm.abc import AbstractErgm, AbstractSufficientStatistics

__all__ = ("AbstractErgmFunctions",)


class AbstractErgmFunctions(AbstractModelFunctions):
    """ERGM model functions container.

    Examples
    --------
    One can check that Hamiltonian and partition function can be used to calculate
    graph probabilities.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(303)
    >>> n = 100
    >>> model = RandomGraph(n, mu=rng.normal(n) - 1.5)
    >>> S = model.sample(rng=rng)
    >>> A = S.A.toarray()
    >>> D = A.sum(axis=1)
    >>> i, j = jnp.tril_indices(n, k=-1)
    >>> P = model.pairs[i, j].probs()
    >>> A = A[i, j]
    >>> loglik1 = jnp.log(P[A == 1]).sum() + jnp.log((1 - P)[A == 0]).sum()
    >>> H = model.hamiltonian(S.A)
    >>> F = model.free_energy()
    >>> loglik2 = -H + F
    >>> jnp.isclose(loglik1, loglik2).item()
    True
    """

    @classmethod
    @eqx.filter_jit
    def free_energy(cls, model: "AbstractErgm", *args: Any, **kwargs: Any) -> Reals:
        """Compute the free energy of the model."""
        raise NotImplementedError

    @classmethod
    @eqx.filter_jit
    def partition_function(
        cls, model: "AbstractErgm", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the partition function of the model."""
        free_energy = cls.free_energy(model, *args, **kwargs)
        return jnp.exp(-free_energy)

    @classmethod
    def hamiltonian(cls, model: "AbstractErgm", obj: tuple, **kwargs: Any) -> Real:
        """Compute the Hamiltonian of the model."""
        fit = model.fit(obj, **kwargs)
        return cls._hamiltonian(model, fit.target)

    @classmethod
    def lagrangian(cls, model: "AbstractErgm", obj: tuple, **kwargs: Any) -> Real:
        """Compute the Lagrangian of the model."""
        fit = model.fit(obj, **kwargs)
        return cls._lagrangian(model, fit.target)

    # Internals ----------------------------------------------------------------------

    @classmethod
    @eqx.filter_jit
    def _hamiltonian(
        cls, model: "AbstractErgm", stats: "AbstractSufficientStatistics"
    ) -> Real:
        return model.fit(stats).hamiltonian()

    @classmethod
    @eqx.filter_jit
    def _lagrangian(
        cls, model: "AbstractErgm", stats: "AbstractSufficientStatistics"
    ) -> Real:
        H = cls._hamiltonian(model, stats)
        F = model.free_energy()
        return H - F
