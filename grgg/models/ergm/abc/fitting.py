from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping
from inspect import isabstract
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from grgg._typing import Numbers, Real
from grgg.models.abc import AbstractFitTarget, AbstractModelFit
from grgg.utils.misc import split_kwargs

if TYPE_CHECKING:
    from grgg.models.ergm.abc import AbstractErgmModel

__all__ = (
    "AbstractObservedStatistics",
    "AbstractErgmFit",
    "AbstractSufficientStatistics",
    "LagrangianErgmFit",
)


T = TypeVar("T", bound="AbstractErgmModel")
ObjectiveT = Callable[[T, ...], Real]


class AbstractObservedStatistics(AbstractFitTarget[Numbers]):
    """Abstract base class for observed statistics used in ERGM fitting."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not isabstract(cls):
            mapping = cls.get_stats2params()
            if set(mapping.keys()) != set(cls.names):
                errmsg = (
                    f"Statistic names {cls.names}"
                    f" do not match mapping keys {list(mapping)}"
                )
                raise TypeError(errmsg)

    @property
    def stats2params(self) -> Mapping[str, str]:
        """Mapping from statistic names to their corresponding parameter names."""
        return self.get_stats2params()

    @property
    def params2stats(self) -> Mapping[str, str]:
        """Mapping from parameter names to their corresponding statistic names."""
        return {v: k for k, v in self.stats2params.items()}

    @classmethod
    @abstractmethod
    def get_stats2params(cls) -> Mapping[str, str]:
        """Return a mapping from statistic names to parameter names."""


class AbstractErgmFit[T](AbstractModelFit[T]):
    """Abstract base class for ERGM model fitting procedures."""

    target: AbstractObservedStatistics

    @property
    def stats(self) -> AbstractObservedStatistics:
        """Observed statistics used for fitting (alias for `self.target`)."""
        return self.target

    @classmethod
    def make_target(
        cls,
        model: T,
        data: Any,
        *args: Any,
        homogeneous: bool | str | Iterable[str] | None = None,
        heterogeneous: bool | str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> AbstractObservedStatistics:
        """Convert raw data into a target format suitable for ERGM fitting."""
        # Get initial target stats
        target = super().make_target(model, data, *args, **kwargs)
        homogeneous = cls._get_homogeneous_names(
            model, target, homogeneous, heterogeneous
        )
        stats = {
            name: cls.postprocess_statistic(target[name], name in homogeneous)
            for name in target.names
        }
        return target.replace(**stats)

    @classmethod
    def _get_homogeneous_names(
        cls,
        model: T,
        target: AbstractObservedStatistics,
        homogeneous: bool | str | Iterable[str] | None = None,
        heterogeneous: bool | str | Iterable[str] | None = None,
    ) -> None:
        if (homogeneous, heterogeneous) in ((True, True), (False, False)):
            errmsg = "'homogeneous' and 'heterogeneous' cannot be both True/False"
            raise ValueError(errmsg)
        if homogeneous is None and heterogeneous is None:
            homogeneous = set()
            for name, param in model.parameters.to_dict().items():
                if param.is_homogeneous:
                    homogeneous.add(target.params2stats[name])
            return homogeneous

        if homogeneous is None:
            homogeneous = heterogeneous is not True
        if heterogeneous is None:
            heterogeneous = homogeneous is not True

        heterogeneous, homogeneous = (
            set(target.names)
            if x is True
            else set()
            if x is False
            else {x}
            if isinstance(x, str)
            else set(x)
            for x in (heterogeneous, homogeneous)
        )
        if heterogeneous & homogeneous:
            errmsg = (
                f"parameters {heterogeneous & homogeneous} "
                "cannot be both homogeneous and heterogeneous"
            )
            raise ValueError(errmsg)
        return homogeneous

    @classmethod
    @abstractmethod
    def postprocess_statistic(cls, statistic: Numbers, homogeneous: bool) -> ArrayLike:
        """Postprocess a statistic based on its corresponding parameter."""

    def compute_expectations(
        self,
        model: T | None = None,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> AbstractObservedStatistics:
        """Compute expected statistics."""
        model = model if model is not None else self.model
        options = options or {}
        expectations = {}
        for name in self.target.names:
            stat = model.get_statistic(name)
            opts, _ = split_kwargs(
                stat.get_instance_fields(), **{**options, **stats_options.get(name, {})}
            )
            expectations[name] = stat(**opts)
        return self.target.replace(**expectations)

    def postprocess_expectations(
        self,
        expectations: AbstractObservedStatistics,
        model: T | None = None,
    ) -> AbstractObservedStatistics:
        """Postprocess expected statistics after fitting."""
        model = model if model is not None else self.model
        stats = {
            name: self.postprocess_statistic(
                expectation,
                self.model.params[self.target.stats2params[name]].shape == (),
            )
            for name, expectation in expectations.to_dict().items()
        }
        return expectations.replace(**stats)


class AbstractSufficientStatistics(AbstractObservedStatistics):
    """Abstract base class for sufficient statistics used in ERGM fitting."""


class LagrangianErgmFit[T](AbstractErgmFit[T]):
    """ERGM fitter based on the model Lagrangian."""

    target: AbstractSufficientStatistics
    alias: ClassVar[str] = "lagrangian"

    @property
    def sufficient_statistics(self) -> AbstractSufficientStatistics:
        """Sufficient statistics used for fitting (alias for `self.target`)."""
        return self.target

    @classmethod
    def postprocess_statistic(cls, statistic: Numbers, homogeneous: bool) -> Numbers:
        if homogeneous:
            statistic = statistic.sum()
        return statistic

    def hamiltonian(self) -> Real:
        """Compute the Hamiltonian of the model."""
        stats = self.sufficient_statistics
        H = 0.0
        for name, stat in stats.to_dict().items():
            param = self.model.parameters[stats.stats2params[name]]
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

    def define_objective(
        self,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> Real:
        """Define Lagrangian objective function.

        Examples
        --------
        >>> import jax.numpy as jnp  # doctest: +SKIP
        >>> import igraph as ig  # doctest: +SKIP
        >>> from grgg import RandomGenerator, RandomGraph  # doctest: +SKIP
        >>> rng = RandomGenerator(0)  # doctest: +SKIP
        >>> n = 1000  # doctest: +SKIP
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)  # doctest: +SKIP
        >>> G = ig.Graph.Erdos_Renyi(n, p=10/n)  # doctest: +SKIP
        >>> fit = model.fit(G)  # doctest: +SKIP
        >>> objective = fit.define_objective()  # doctest: +SKIP
        >>> grad = jax.grad(objective)(model)  # doctest: +SKIP
        >>> def lagrangian(model):  # doctest: +SKIP
        ...     return model.lagrangian(fit.stats)
        >>> grad_naive = jax.grad(lagrangian)(model)  # doctest: +SKIP
        >>> g, g_naive = grad.mu.data, grad_naive.mu.data  # doctest: +SKIP
        >>> jnp.allclose(g, g_naive, rtol=1e-3).item()  # doctest: +SKIP
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
            expectations = self.compute_expectations(model, options, **stats_options)
            expectations = self.postprocess_expectations(expectations, model)
            gradient = model.parameters.__class__(
                *jax.tree.map(lambda e, s: (e - s) * g_out, expectations, self.target)
            )
            return model.replace(parameters=gradient)

        return objective
