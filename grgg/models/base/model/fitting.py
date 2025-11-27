from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from grgg._typing import Real
from grgg.models.base.observables import AbstractObservables
from grgg.utils.dispatch import dispatch
from grgg.utils.dotpath import dotget
from grgg.utils.misc import split_kwargs

from .modules import AbstractModelModule

__all__ = ("ModelFit",)


T = TypeVar("T", bound="AbstractModel")
O = TypeVar("O", bound="AbstractObservables")


class ModelFit[T, O](AbstractModelModule[T]):
    """Abstract base class for model fitting procedures."""

    model: T
    target: O

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.model.equals(other.model)
            and self.target.equals(other.target)
        )

    @dispatch
    def define_objective(self, **kwargs: Any) -> "ObjectiveT":
        """Define the objective function for fitting."""
        return self.define_objective(self.target, **kwargs)

    def compute_expectations(
        self,
        model: T | None = None,
        *,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> AbstractObservables:
        """Compute expected statistics.

        Parameters
        ----------
        model
            The model to compute expectations for. If `None`, uses `self.model`.
        options
            General options to pass to all statistic methods.
        **stats_options
            Specific options to pass to individual statistic methods.
        """
        model = model if model is not None else self.model
        options = options or {}
        expectations = {}
        for name, stat in self.target.to_dict().items():
            statmethod = dotget(
                self.model, self.target.fields[name].metadata["statistic"]
            )
            opts, _ = split_kwargs(
                statmethod.get_instance_fields(),
                **{**options, **stats_options.get(name, {})},
            )
            expectation = statmethod(**opts)
            if stat.is_scalar:
                expectation = self.target.reduce(expectation)
            expectations[name] = expectation
        return self.target.replace(**expectations)


# Avoid circular imports -------------------------------------------------------------

from .model import AbstractModel  # noqa

ObjectiveT = Callable[[AbstractModel, ...], Real]

# @classmethod
# def _make_target(
#     cls,
#     model: T,
#     data: Any,
#     *args: Any,
#     homogeneous: bool | str | Iterable[str] | None = None,
#     heterogeneous: bool | str | Iterable[str] | None = None,
#     **kwargs: Any,
# ) -> S:
#     """Convert raw data into a target format suitable for ERGM fitting.

#     Examples
#     --------
#     >>> import jax.numpy as jnp
#     >>> from grgg import RandomGraph
#     >>> n = 100
#     >>> model = RandomGraph(n)
#     >>> A = jnp.ones((n, n)) - jnp.eye(n)
#     >>> fit = model.fit(A)
#     >>> fit.target
#     RandomGraphSufficientStatistics(degree=f...[100])
#     >>> fit = model.fit(A, heterogeneous=True)
#     >>> fit.target
#     RandomGraphSufficientStatistics(degree=f...[100])
#     >>> fit = model.fit(A, homogeneous=False)
#     >>> fit.target
#     RandomGraphSufficientStatistics(degree=f...[100])
#     >>> model = RandomGraph(n, mu=jnp.linspace(-3, 3, n))
#     >>> fit = model.fit(A)
#     >>> fit.target
#     RandomGraphSufficientStatistics(degree=f...[100])
#     >>> fit = model.fit(A, homogeneous=True)
#     >>> fit.target
#     RandomGraphSufficientStatistics(degree=9900.00)
#     >>> fit = model.fit(A, homogeneous="degree")
#     >>> fit.target
#     RandomGraphSufficientStatistics(degree=9900.00)

#     Determination of fit target is idempotent.
#     >>> fit2 = model.fit(fit.target)
#     >>> fit2.equals(fit)
#     True

#     Creation from explicit statistics.
#     >>> model.fit(degree=10).target
#     RandomGraphSufficientStatistics(degree=10)

#     Check hamiltonian calculations.
#     >>> model = RandomGraph(n)
#     >>> fit = model.fit(A)
#     >>> fit.hamiltonian().item()
#     0.0
#     >>> model = RandomGraph(n, mu=-2)
#     >>> fit = model.fit(A)
#     >>> fit.hamiltonian().item()
#     9900.0
#     >>> fit.hamiltonian().item() == model.hamiltonian(A).item()
#     True

#     Check lagrangian calculations.
#     >>> fit.lagrangian().item() == model.lagrangian(A).item()
#     True
#     """
#     target = cls.make_target(model, data, *args, **kwargs)
#     if homogeneous is None and heterogeneous is None or homogeneous is False:
#         heterogeneous = True
#     if heterogeneous is False and homogeneous is None:
#         homogeneous = True
#     _homogeneous, _ = partition_choices(
#         target.names, homogeneous=homogeneous, heterogeneous=heterogeneous
#     )
#     stats = {
#         name: target.reduce(stat) if name in _homogeneous else stat
#         for name, stat in target.to_dict().items()
#     }
#     return target.replace(**stats)
