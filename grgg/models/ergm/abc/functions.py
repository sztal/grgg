from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Real, Reals
from grgg.models.abc import AbstractModelFunctions

if TYPE_CHECKING:
    from grgg.models.ergm.abc import AbstractErgm

__all__ = ("AbstractErgmFunctions",)


LagrangianT = Callable[..., Real]


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

    @singledispatchmethod
    @classmethod
    def hamiltonian(cls, obj: Any, model, **kwargs: Any) -> Real:
        """Compute the Hamiltonian of the model."""
        stats = model.sufficient_statistics(obj, **kwargs)
        return cls.hamiltonian(stats, model, **kwargs)

    @hamiltonian.register
    @classmethod
    @eqx.filter_jit
    def _(cls, stats: tuple, model) -> Real:
        return _hamiltonian(model, stats)

    @singledispatchmethod
    @classmethod
    def define_lagrangian(cls, obj: Any, model, **kwargs: Any) -> LagrangianT:
        """Define the Lagrangian function for the model given an object."""
        stats = model.sufficient_statistics(obj, **kwargs)
        return cls.define_lagrangian(stats, model, **kwargs)

    @define_lagrangian.register
    @classmethod
    def _(cls, stats: tuple, model) -> LagrangianT:  # noqa
        @jax.jit
        def lagrangian(model: "AbstractErgm") -> Real:
            H = model.hamiltonian(stats)
            F = model.free_energy()
            return H - F

        return lagrangian


# Internals --------------------------------------------------------------------------


def _hamiltonian(model: "AbstractErgm", stats: tuple) -> Real:
    H = jax.new_ref(0.0)
    params = model.parameters
    for i, _ in enumerate(model.Parameters.names):
        stat = stats[i]
        param = params[i]
        H_i = jnp.sum(param.theta * stat)  # 'param.theta' is Lagrange multiplier
        if not model.is_directed and model.is_homogeneous:
            H_i /= 2
        H[...] += H_i
    return H[...]
