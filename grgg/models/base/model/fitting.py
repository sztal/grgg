from abc import abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, TypeVar

import equinox as eqx

from grgg._typing import ObjectiveFunction
from grgg.utils.misc import split_kwargs
from grgg.utils.variables import ArrayBundle

from .modules import AbstractModelModule

__all__ = ("AbstractModelFit", "LeastSquaresFit")


T = TypeVar("T", bound="AbstractModel")
O = TypeVar("O", bound="ArrayBundle")


class AbstractModelFit[T, O](AbstractModelModule[T]):
    """Abstract base class for model fitting procedures."""

    model: eqx.AbstractVar[T]
    target: eqx.AbstractVar[O]

    is_fitted: bool = eqx.field(default=False, static=True, kw_only=True)
    is_initialized: bool = eqx.field(default=False, static=True, kw_only=True)

    method: eqx.AbstractClassVar[str]

    def __check_init__(self) -> None:
        expected_stat_names = [
            p.get_statistic(self.model, self.method, homogeneous=False)[0]
            for p in self.model.parameters
        ]
        missing_stats = set(expected_stat_names) - set(self.target.mapping.keys())
        if missing_stats:
            errmsg = (
                f"target is missing required statistics: "
                f"{', '.join(repr(s) for s in missing_stats)}."
            )
            raise ValueError(errmsg)

    @abstractmethod
    def define_objective(self, *args: Any, **kwargs: Any) -> ObjectiveFunction:
        """Define the objective function for fitting."""

    def compute_expectations(
        self,
        model: T | None = None,
        *,
        normalize: bool = False,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> ArrayBundle:
        """Compute expected target values.

        Parameters
        ----------
        model
            The model to compute expectations for. If `None`, uses `self.model`.
        normalize
            Whether to normalize the expectations by the number of units.
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
                **{**options, **stats_options.get(name, {}), "normalize": normalize},
            )
            expectation = method(**opts)
            expectations[stat] = expectation
        return ArrayBundle(**expectations)

    def get_expected_statistics_names(self) -> list[str]:
        """Get the names of the expected statistics used in fitting."""
        return [
            param.get_statistic(self.model, self.method)[0]
            for param in self.model.parameters
        ]


class LeastSquaresFit(AbstractModelFit[T, O]):
    """Least squares fitting procedure."""

    model: T
    target: O
    method: ClassVar[str] = "least_squares"


# Avoid circular imports -------------------------------------------------------------

from .model import AbstractModel  # noqa
