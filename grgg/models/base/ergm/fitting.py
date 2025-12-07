from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from grgg._typing import Number, Numbers, Real
from grgg.utils.dispatch import dispatch
from grgg.utils.variables import ArrayBundle

from ..model import AbstractModelFit

__all__ = (
    "SufficientStatistics",
    "ExpectedStatistics",
    "LagrangianFit",
)


T = TypeVar("T", bound="AbstractErgm")
FT = TypeVar("FT", bound="AbstractErgmFitTarget")
SS = TypeVar("SS", bound="SufficientStatistics")
OS = TypeVar("OS", bound="ExpectedStatistics")


class AbstractErgmFitTarget(ArrayBundle[Numbers]):
    """Abstract base class for ERGM fitting targets."""

    reduction: eqx.AbstractClassVar[Literal["sum", "mean"]]

    def reduce(self, statistic: ArrayLike) -> Number:
        """Reduce statistic according to the specified reduction method."""
        return self._reduce(self.reduction, statistic)

    @dispatch.abstract
    def _reduce(self, _: Literal["sum", "mean"], statistic: Any) -> Number:
        """Implementation of reduction methods."""

    @_reduce.dispatch
    @eqx.filter_jit
    def _(self, _: Literal["sum"], statistic: Any) -> Number:
        return jnp.sum(statistic)

    @_reduce.dispatch
    @eqx.filter_jit
    def _(self, _: Literal["mean"], statistic: Any) -> Number:
        return jnp.mean(statistic)


class ExpectedStatistics(AbstractErgmFitTarget):
    """Abstract base class for expected statistics used in ERGM fitting."""

    reduction: ClassVar[Literal["mean"]] = "mean"


class SufficientStatistics(AbstractErgmFitTarget):
    """Abstract base class for sufficient statistics used in ERGM fitting."""

    reduction: ClassVar[Literal["sum"]] = "sum"


class LagrangianFit[T, SS](AbstractModelFit[T, SS]):
    """ERGM model fit based on the model Lagrangian."""

    target: SufficientStatistics
    method: ClassVar[str] = "lagrangian"

    @property
    def sufficient_statistics(self) -> SufficientStatistics:
        """Sufficient statistics used for fitting (alias for `self.target`)."""
        return self.target

    def hamiltonian(self) -> Real:
        """Compute the Hamiltonian of the model.

        Examples
        --------
        >>> import random
        >>> import igraph as ig
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGenerator, RandomGraph
        >>> random.seed(303)
        >>> rng = RandomGenerator(17)
        >>> n = 100
        >>> G = ig.Graph.Erdos_Renyi(n, p=0.1)
        >>> model = RandomGraph(n)

        Homogeneous (Erdős-Rényi) case.
        >>> fit = model.fit(G, homogeneous=True)
        >>> H = fit.hamiltonian()
        >>> H_naive = fit.model.mu.theta * G.ecount()
        >>> jnp.allclose(H, H_naive).item()
        True

        Heterogeneous (soft configuration model) case.
        >>> fit = model.fit(G, heterogeneous=True)
        >>> H = fit.hamiltonian()
        >>> H_naive = jnp.sum(fit.model.mu.theta * jnp.array(G.degree()))
        >>> jnp.allclose(H, H_naive).item()
        True
        """
        stats = self.sufficient_statistics
        H = 0.0
        for param in self.model.parameters:
            stat, _ = param.get_statistic(self.model, self.method)
            # 'param.theta' is Lagrange multiplier
            H += jnp.sum(param.theta * stats[stat])
        return H

    def lagrangian(self) -> Real:
        """Compute the Lagrangian of the model.

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
        >>> G = ig.Graph.Erdos_Renyi(n, p=10/n)

        Homogeneous (Erdős-Rényi) case.
        >>> fit = RandomGraph(n).fit(G, homogeneous=True)
        >>> objective = fit.define_objective()
        >>> grad = jax.grad(objective)(fit.model)
        >>> def lagrangian(model):
        ...     return model.lagrangian(fit.sufficient_statistics)
        >>> grad_naive = jax.grad(lagrangian)(fit.model)
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data
        >>> jnp.allclose(g, g_naive, rtol=1e-3).item()
        True

        Heterogeneous (soft configuration model) case.
        >>> fit = RandomGraph(n).fit(G, heterogeneous=True)
        >>> objective = fit.define_objective()
        >>> grad = jax.grad(objective)(fit.model)
        >>> def lagrangian(model):
        ...     return model.lagrangian(fit.sufficient_statistics)
        >>> grad_naive = jax.grad(lagrangian)(fit.model)
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data
        >>> jnp.allclose(g, g_naive, rtol=1e-2).item()
        True
        """
        H = self.hamiltonian()
        F = self.model.free_energy()
        return H - F

    @dispatch
    def define_objective(
        self,
        target: SufficientStatistics,  # noqa
    ) -> "ObjectiveT":
        """Define Lagrangian objective function."""
        return self._define_objective()

    def _define_objective(
        self,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> Real:
        """Define Lagrangian objective function."""

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
