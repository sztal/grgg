from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from grgg._typing import Number, Real
from grgg.utils.dispatch import dispatch

from ..model import ModelFit
from ..observables import AbstractObservables

__all__ = (
    "AbstractSufficientStatistics",
    "LagrangianFit",
)


T = TypeVar("T", bound="AbstractErgm")
FT = TypeVar("FT", bound="AbstractErgmFitTarget")
SS = TypeVar("SS", bound="AbstractSufficientStatistics")


class AbstractErgmFitTarget(AbstractObservables):
    """Abstract base class for ERGM fitting targets."""

    reduction: eqx.AbstractClassVar[Literal["sum", "mean"]]

    def reduce(self, statistic: ArrayLike) -> Number:
        """Reduce statistic according to the specified reduction method."""
        return self._reduce_impl(self.reduction, statistic)

    @dispatch.abstract
    def _reduce_impl(self, _: Literal["sum", "mean"], statistic: Any) -> Number:
        """Implementation of reduction methods."""

    @_reduce_impl.dispatch
    @eqx.filter_jit
    def _(self, _: Literal["sum"], statistic: Any) -> Number:
        return jnp.sum(statistic)

    @_reduce_impl.dispatch
    @eqx.filter_jit
    def _(self, _: Literal["mean"], statistic: Any) -> Number:
        return jnp.mean(statistic)


class AbstractExpectedStatistics(AbstractErgmFitTarget):
    """Abstract base class for expected statistics used in ERGM fitting."""

    reduction: ClassVar[Literal["mean"]] = "mean"


class AbstractSufficientStatistics(AbstractErgmFitTarget):
    """Abstract base class for sufficient statistics used in ERGM fitting."""

    reduction: ClassVar[Literal["sum"]] = "sum"


class LagrangianFit[T, SS](ModelFit[T, SS]):
    """ERGM model fit based on the model Lagrangian."""

    target: AbstractSufficientStatistics
    alias: ClassVar[str] = "lagrangian"

    @property
    def sufficient_statistics(self) -> AbstractSufficientStatistics:
        """Sufficient statistics used for fitting (alias for `self.target`)."""
        return self.target

    def hamiltonian(self) -> Real:
        """Compute the Hamiltonian of the model."""
        stats = self.sufficient_statistics
        H = 0.0
        for name, stat in stats.to_dict().items():
            meta = stats.fields[name].metadata
            param = self.model.parameters[meta["parameter"]]
            H_i = jnp.sum(param.theta * stat)  # 'param.theta' is Lagrange multiplier
            if not self.model.is_directed and self.model.is_homogeneous:
                H_i /= 2
            H += H_i
        return H

    def lagrangian(self) -> Real:
        """Compute the Lagrangian of the model."""
        H = self.hamiltonian()
        F = self.model.free_energy()
        return H - F

    @dispatch
    def define_objective(
        self,
        target: AbstractSufficientStatistics,  # noqa
    ) -> "ObjectiveT":
        """Define Lagrangian objective function."""
        return self._define_objective_impl()

    def _define_objective_impl(
        self,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> Real:
        """Define Lagrangian objective function.

        Examples
        --------
        >>> import random
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import igraph as ig
        >>> from grgg import RandomGenerator, RandomGraph
        >>> random.seed(17)
        >>> rng = RandomGenerator(0)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> G = ig.Graph.Erdos_Renyi(n, p=10/n)
        >>> fit = model.fit(G)
        >>> objective = fit.define_objective()
        >>> grad = jax.grad(objective)(model)
        >>> def lagrangian(model):
        ...     return model.lagrangian(fit.sufficient_statistics)
        >>> grad_naive = jax.grad(lagrangian)(model)
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data
        >>> jnp.allclose(g, g_naive, rtol=1e-3).item()
        True
        """

        @eqx.filter_custom_vjp
        @eqx.filter_jit
        def objective(model: T) -> Real:
            return model.lagrangian(self.target)

        @objective.def_fwd
        @eqx.filter_jit
        def objective_fwd(_, model: T) -> tuple[Real, None]:
            return objective(model), None

        @objective.def_bwd
        @eqx.filter_jit
        def objective_bwd(_, g_out: Real, __, model: T) -> T:
            expectations = self.compute_expectations(
                model, options=options, **stats_options
            )
            gradient = model.parameters.__class__(
                *jax.tree.map(lambda e, s: (e - s) * g_out, expectations, self.target)
            )
            return model.replace(parameters=gradient)

        return objective


# Avoid circular imports -------------------------------------------------------------

from .model import AbstractErgm  # noqa

ObjectiveT = Callable[[AbstractErgm, ...], Real]
