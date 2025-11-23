from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import singledispatchmethod
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import DTypeLike

from grgg._typing import Number, Numbers
from grgg.abc import AbstractModule
from grgg.utils.indexing import Shaped
from grgg.utils.misc import format_array

__all__ = ("AbstractParameter", "AbstractParameters", "Constraints")


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
    eqx.error_if(parameter, is_bad, f"'{parameter.name}' must be {constraint}")


class AbstractParameter(Shaped):
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

    data: Numbers
    name: str = eqx.field(static=True, repr=False)

    constraints: eqx.AbstractClassVar[tuple[Constraints, ...]]
    default_value: eqx.AbstractClassVar[Number]

    def __init__(
        self,
        data: jnp.ndarray | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.data = jnp.asarray(data if data is not None else self.default_value)
        if not eqx.is_inexact_array(self.data):
            self.data = self.data.astype(float)
        self.name = name or self.__class__.__name__.lower()

    def __jax_array__(self) -> Numbers:
        return self.data

    def __check_init__(self):
        for constraint in self.constraints:
            _validate(self, constraint.value)

    def __repr__(self) -> str:
        if not isinstance(self.data, jnp.ndarray):
            inner = self.data
        else:
            inner = self.data.item() if self.is_scalar else format_array(self.data)
        return f"{self.__class__.__name__}({inner})"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        return self.replace(data=self.data[args])

    def __len__(self) -> int:
        return len(self.data)

    def __pos__(self) -> Self:
        return self.replace(data=+self.data)

    def __neg__(self) -> Self:
        return self.replace(data=-self.data)

    def __add__(self, other: Numbers) -> Self:
        if isinstance(other, type(self)):
            return self.replace(data=self.data + other.data)
        if isinstance(other, AbstractParameter):
            return jnp.add(self, other)
        return self.replace(data=self.data + other)

    def __sub__(self, other: Numbers) -> Self:
        if isinstance(other, type(self)):
            return self.replace(data=self.data - other.data)
        if isinstance(other, AbstractParameter):
            return jnp.subtract(self, other)
        return self.replace(data=self.data - other)

    def __mul__(self, other: Numbers) -> Self:
        if isinstance(other, type(self)):
            return self.replace(data=self.data * other.data)
        if isinstance(other, AbstractParameter):
            return jnp.multiply(self, other)
        return self.replace(data=self.data * other)

    def __div__(self, other: Numbers) -> Self:
        if isinstance(other, type(self)):
            return self.replace(data=self.data / other.data)
        if isinstance(other, AbstractParameter):
            return jnp.divide(self, other)
        return self.replace(data=self.data / other)

    @property
    def value(self) -> jnp.ndarray:
        """Parameter value(s)."""
        return self.data

    @property
    @abstractmethod
    def theta(self) -> jnp.ndarray:
        """Raw Lagrange multiplier representation of the parameter."""

    @property
    def dtype(self) -> DTypeLike:
        """Data type of the parameter data."""
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameter data."""
        return self.data.shape

    @property
    def is_scalar(self) -> bool:
        """Whether the parameter is a scalar."""
        return hasattr(self.data, "ndim") and self.data.ndim == 0

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

    def astype(self, dtype: DTypeLike) -> Self:
        return self.replace(data=self.data.astype(dtype))


class AbstractParameters(AbstractModule, Sequence[AbstractParameter]):
    """Abstract base class for model parameters container."""

    names: eqx.AbstractClassVar[tuple[str, ...]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        names = list(getattr(cls, "names", []))
        for param, typ in cls.__annotations__.items():
            if issubclass(typ, AbstractParameter) and param not in names:
                names.append(param)
        cls.names = tuple(names)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__repr_inner__()})"

    def __repr_inner__(self) -> str:
        params = [getattr(self, name) for name in self.names]
        return ", ".join(map(repr, params))

    @property
    def are_heterogeneous(self) -> bool:
        """Whether the parameters container has heterogeneous parameters."""
        return any(param.is_heterogeneous for param in self)

    @property
    def are_homogeneous(self) -> bool:
        """Whether the parameters container has homogeneous parameters."""
        return not self.are_heterogeneous

    @property
    def dtype(self) -> DTypeLike:
        """Resolved parameter data type."""
        if not self:
            errmsg = "cannot determine dtype of empty parameters' set"
            raise ValueError(errmsg)
        dtype = self[0].dtype
        for i in range(1, len(self)):
            param = self[i]
            dtype = jnp.promote_types(dtype, param.dtype)
        return dtype

    def __len__(self) -> int:
        return len(self.names)

    @singledispatchmethod
    def __getitem__(self, index: Any) -> AbstractParameter:
        errmsg = f"index must be 'int' or 'str', got '{type(index).__name__}'"
        raise TypeError(errmsg)

    @__getitem__.register
    def _(self, index: int) -> AbstractParameter:
        return getattr(self, self.names[index])

    @__getitem__.register
    def _(self, name: str) -> AbstractParameter:
        if name not in self.names:
            errmsg = f"unknown parameter name '{name}'"
            raise KeyError(errmsg)
        return getattr(self, name)

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and all(
            jnp.array_equal(getattr(self, name).data, getattr(other, name).data)
            for name in self.names
        )

    def resolve_dtype(self) -> Self:
        """Return `self` with resolved dtypes."""
        dtype = self.dtype
        return self.__class__(*(param.astype(dtype) for param in self))
