from collections.abc import Callable
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import equinox as eqx
from jaxtyping import PyTree

from grgg._typing import Real
from grgg.models.abc import AbstractModelOptimizer

if TYPE_CHECKING:
    from .model import AbstractErgm

__all__ = ("ErgmOptimizer",)


T = TypeVar("T", bound="AbstractErgm")

LagrangianT = Callable[[T, ...], Real]
ObjectiveT = Literal["lagrangian", "least_squares", "fixed_point"]


class ErgmOptimizer[T](AbstractModelOptimizer[T]):
    """Base ERGM optimizer."""

    def optimize(
        self, *args: Any, objective: ObjectiveT = "lagrangian", **kwargs: Any
    ) -> T:
        """Optimize the model parameters."""
        try:
            method = getattr(self, f"optimize_{objective}")
        except AttributeError as exc:
            errmsg = f"unknown optimization objective '{objective}'"
            raise ValueError(errmsg) from exc
        return method(*args, **kwargs)

    @singledispatchmethod
    def define_lagrangian(self, obj: Any, *args: Any, **kwargs: Any) -> LagrangianT:
        """Define the Lagrangian of the model given an object
        for which sufficient statistics can be computed.

        Examples
        --------
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> n = 1000
        >>> rng = RandomGenerator(303)
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> S = model.sample(rng=rng)
        >>> lagrangian = model.define_lagrangian(S.A)
        >>> nll1 = lagrangian(model)  # negative log-likelihood
        >>> nll2 = model.hamiltonian(S.A) - model.free_energy()
        >>> jnp.isclose(nll1, nll2).item()
        True

        Lagrangian function can be differentiated w.r.t. model parameters
        >>> jax.grad(lagrangian)(model).mu
        Mu(...[1000])
        """
        stats = self.model.sufficient_statistics(obj, *args, **kwargs)
        return self.define_lagrangian(stats)

    @define_lagrangian.register
    def _(self, stats: tuple) -> LagrangianT:
        @eqx.filter_jit
        def lagrangian(model: T, args: PyTree[Any] = None) -> Real:
            args = (args,) if args else ()
            return model.functions._lagrangian(model, stats, *args)

        return lagrangian
