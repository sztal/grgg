from collections.abc import Mapping
from enum import Enum
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from grgg.abc import AbstractModule
from grgg.utils.misc import format_array

__all__ = ("AbstractParameter", "Constraints", "ParametersMapping")


class Constraints(Enum):
    """Enumeration of parameter constraints."""

    REAL = "real"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NON_POSITIVE = "non-positive"
    NON_NEGATIVE = "non-negative"


def _validate(parameter: "AbstractParameter", constraint: str) -> None:
    is_bad = False
    constraint = constraint.lower()
    if constraint == Constraints.REAL.value:
        is_bad = ~jnp.isreal(parameter.data).all()
    elif constraint == Constraints.POSITIVE.value:
        is_bad = ~jnp.all(parameter.data > 0)
    elif constraint == Constraints.NEGATIVE.value:
        is_bad = ~jnp.all(parameter.data < 0)
    elif constraint == Constraints.NON_POSITIVE.value:
        is_bad = ~jnp.all(parameter.data <= 0)
    elif constraint == Constraints.NON_NEGATIVE.value:
        is_bad = ~jnp.all(parameter.data >= 0)
    else:
        errmsg = f"unknown constraint '{constraint}'"
        raise ValueError(errmsg)
    if is_bad:
        errmsg = f"'{parameter.name}' must be {constraint}"
        raise ValueError(errmsg)


class AbstractParameter(AbstractModule):
    """Abstract base class for model parameters.

    Units are laid out along the first axis.
    The rest of the shape is the parameter's internal shape.

    Attributes
    ----------
    data
        Parameter value(s).
    constraints
        Constraints on the parameter value(s).
    """

    data: jnp.ndarray
    name: str = eqx.field(static=True, repr=False)

    ndims: eqx.AbstractClassVar[tuple[int, ...]]
    constraints: eqx.AbstractClassVar[tuple[Constraints, ...]]
    default_value: eqx.AbstractClassVar[float]

    def __init__(
        self, data: jnp.ndarray | None = None, name: str | None = None
    ) -> None:
        data = jnp.asarray(data if data is not None else self.default_value)
        if jnp.issubdtype(data.dtype, jnp.integer):
            data = data.astype(float)
        self.data = data
        self.name = name or self.__class__.__name__.lower()

    def __check_init__(self):
        if self.data.ndim not in self.ndims:
            errmsg = f"'data.ndim' must be in {self.ndims}, got {self.data.ndim}"
            raise ValueError(errmsg)
        for constraint in self.constraints:
            _validate(self, constraint.value)

    def __repr__(self) -> str:
        if self.is_scalar:
            return f"{self.__class__.__name__}({self.data.item()})"
        return f"{self.__class__.__name__}({format_array(self.data)})"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        return self.replace(data=self.data[args])

    def __len__(self) -> int:
        return len(self.data)

    @property
    def value(self) -> jnp.ndarray:
        """Parameter value(s)."""
        return self.data

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameter data."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the parameter data."""
        return self.data.ndim

    @property
    def size(self) -> int:
        """Number of elements in the parameter data."""
        return self.data.size

    @property
    def is_scalar(self) -> bool:
        """Whether the parameter is a scalar."""
        return self.data.ndim == 0

    @property
    def is_homogeneous(self) -> bool:
        """Whether the parameter is homogeneous (all values identical)."""
        return self.is_scalar

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (not all values identical)."""
        return not self.is_homogeneous

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.name == other.name
            and self.constraints == other.constraints
            and jnp.array_equal(self.data, other.data)
        )


class ParametersMapping(Mapping[str, AbstractParameter]):
    """Mapping of model parameters."""

    def __init__(self, parameters: Mapping[str, AbstractParameter]) -> None:
        self._parameters = parameters

    def __getitem__(self, key: str) -> AbstractParameter:
        return self._parameters[key]

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._parameters)})"

    @property
    def data(self) -> Mapping[str, jnp.ndarray]:
        """Mapping of parameter data."""
        return {name: param.data for name, param in self.items()}
