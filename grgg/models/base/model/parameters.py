from abc import abstractmethod
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Real, Reals
from grgg.utils.variables import ArrayBundle, Constraints, Variable

__all__ = ("AbstractParameter", "AbstractParameters")


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


class AbstractParameters(ArrayBundle[AbstractParameter]):
    """Abstract base class for model parameters container."""

    @property
    def are_heterogeneous(self) -> bool:
        """Whether the parameters container has heterogeneous parameters."""
        return any(param.is_heterogeneous for param in self)

    @property
    def are_homogeneous(self) -> bool:
        """Whether the parameters container has homogeneous parameters."""
        return not self.are_heterogeneous
