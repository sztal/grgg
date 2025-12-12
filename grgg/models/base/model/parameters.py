from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, Self

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Numbers, Real, Reals
from grgg.utils.dispatch import dispatch
from grgg.utils.variables import (
    ArrayBundle,
    Constraints,
    FixedArrayBundle,
    Variable,
)

__all__ = ("AbstractParameter", "AbstractParameters", "AbstractObservables")


class AbstractParameter(Variable):
    """Abstract base class for model parameters.

    Attributes
    ----------
    data
        Parameter value(s).
    """

    data: Reals
    constraints: ClassVar[Constraints] = Constraints("real")
    default_value: eqx.AbstractClassVar[Real | float]

    def __init__(self, data: jnp.ndarray | None = None, **kwargs: Any) -> None:
        data = jnp.asarray(data if data is not None else self.default_value)
        if not eqx.is_inexact_array(data):
            data = data.astype(float)
        super().__init__(data, **kwargs)

    @property
    @abstractmethod
    def theta(self) -> jnp.ndarray:
        """Raw Lagrange multiplier representation of the parameter."""

    @property
    def is_homogeneous(self) -> bool:
        """Whether the parameter is homogeneous (all values identical)."""
        return self.is_scalar

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (not all values identical)."""
        return not self.is_homogeneous

    def get_statistic(
        self, model: Any, method: str, *, homogeneous: bool | str | None = None
    ) -> tuple[str, Callable[..., Numbers]]:
        """Get observable statistics corresponding to the parameter.

        Parameters
        ----------
        model
            The model to get statistics from.
        method
            The fitting method to use.
        homogeneous
            Whether to get homogeneous statistics.
            Defaults to the parameter's homogeneity.

        Returns
        -------
        name, statmethod
            The name and statistic method corresponding to the parameter.
        """
        if homogeneous is None:
            homogeneous = self.is_homogeneous
        return self._get_statistic(model, homogeneous, method)

    @dispatch.abstract
    def _get_statistic(
        self,
        model: Any,  # noqa
        homogeneous: bool,  # noqa
        method: str,  # noqa
    ) -> tuple[str, Callable[..., Numbers]]:
        """Get observable statistics corresponding to the parameter."""

    @dispatch.abstract
    def initialize(
        self,
        model: Any,
        target: ArrayBundle,
        method: str,
    ) -> Self:
        """Initialize parameter value(s) from `model` and `target` using `method`."""


class AbstractParameters(FixedArrayBundle[AbstractParameter]):
    """Model parameters container."""

    @property
    def are_heterogeneous(self) -> bool:
        """Whether the parameters container has heterogeneous parameters."""
        return any(param.is_heterogeneous for param in self)

    @property
    def are_homogeneous(self) -> bool:
        """Whether the parameters container has homogeneous parameters."""
        return not self.are_heterogeneous


class AbstractObservables(FixedArrayBundle[Numbers]):
    """Model observables container."""

    def normalize(self, n_units: int) -> Self:
        """Normalize observables by number of units.

        Parameters
        ----------
        n_units
            Number of units to normalize by.

        Returns
        -------
        Normalized observables.
        """
        normalized = {name: self[name] / n_units for name in self.get_instance_fields()}
        return self.__class__(**normalized)
