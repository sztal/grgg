from collections.abc import Callable, Mapping
from typing import Any, ClassVar, TypeVar

import equinox as eqx

from grgg._typing import Real
from grgg.utils.dispatch import dispatch
from grgg.utils.misc import split_kwargs
from grgg.utils.variables import ArrayBundle

from .modules import AbstractModelModule

__all__ = ("AbstractModelFit", "LeastSquaresFit")


T = TypeVar("T", bound="AbstractModel")
O = TypeVar("O", bound="ArrayBundle")


class AbstractModelFit[T, O](AbstractModelModule[T]):
    """Abstract base class for model fitting procedures."""

    model: T
    target: O
    method: eqx.AbstractClassVar[str]

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
    ) -> ArrayBundle:
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
        for name, param in model.parameters.mapping.items():
            stat, method = param.get_statistic(model, self.method)
            opts, _ = split_kwargs(
                method.get_instance_fields(),
                **{**options, **stats_options.get(name, {})},
            )
            expectation = method(**opts)
            expectations[stat] = expectation
        return self.target.replace(**expectations)


class LeastSquaresFit(AbstractModelFit[T, O]):
    """Least squares fitting procedure."""

    method: ClassVar[str] = "least_squares"


# Avoid circular imports -------------------------------------------------------------

from .model import AbstractModel  # noqa

ObjectiveT = Callable[[AbstractModel, ...], Real]
