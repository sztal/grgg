import math
from abc import abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import singledispatchmethod
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg.abc import AbstractModule
from grgg.utils.lazy import LazyOuter
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
    if is_bad:
        errmsg = f"'{parameter.name}' must be {constraint}"
        raise ValueError(errmsg)


class AbstractParameter(AbstractModule, Sequence[jnp.ndarray]):
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

    @property
    def outer(self) -> LazyOuter:
        """Lazy outer sum of the parameter values."""
        return LazyOuter(self.data, self.data, op=jnp.add)

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.name == other.name
            and self.constraints == other.constraints
            and jnp.array_equal(self.data, other.data)
        )


class AbstractParameters(AbstractModule, Sequence[AbstractParameter]):
    """Abstract base class for collections of model parameters.

    Attributes
    ----------
    params
        Model parameters.
    """

    parameters: tuple[AbstractParameter, ...] = eqx.field(converter=tuple)
    definition: eqx.AbstractClassVar[tuple[type[AbstractParameter], ...]]

    def __check_init__(self) -> None:
        if len(self.parameters) != len(self.definition):
            errmsg = f"'parameters' must have length {len(self.definition)}"
            raise ValueError(errmsg)
        for p, d in zip(self.parameters, self.definition, strict=True):
            if not isinstance(p, d):
                errmsg = (
                    f"All parameters must be instances of their respective "
                    f"definition types, expected '{d.__name__}', got "
                    f"'{type(p).__name__}' instead.",
                )
                raise TypeError(errmsg)
        if len(self.names) != len(set(self.names)):
            errmsg = "Parameter names must be unique."
            raise ValueError(errmsg)
        shapes = {p.shape for p in self.parameters if not p.is_scalar}
        if len(shapes) > 1:
            errmsg = (
                "All non-scalar parameters must have the same shape, got "
                f"{', '.join(map(str, shapes))}."
            )
            raise ValueError(errmsg)

    def __repr__(self) -> str:
        inner = ", ".join(f"{p!r}" for p in self.parameters)
        return f"{self.__class__.__name__}({inner})"

    @singledispatchmethod
    def __getitem__(self, args: Any) -> AbstractParameter:
        return self.parameters[args]

    @__getitem__.register
    def _(self, name: str) -> AbstractParameter:
        idx = self.names.index(name.strip().lower())
        return self[idx]

    def __getattr__(self, name: str) -> AbstractParameter:
        try:
            return self[name]
        except ValueError as exc:
            errmsg = f"no parameter named '{name}'"
            raise AttributeError(errmsg) from exc

    def __len__(self) -> int:
        return len(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the parameters."""
        return tuple(p.name for p in self.parameters)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameters."""
        for p in self.parameters:
            if not p.is_scalar:
                return p.shape
        return ()

    @property
    def ndim(self) -> int:
        """Number of dimensions of the parameters."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Number of elements in the parameters."""
        return math.prod(self.shape)

    @property
    def is_heterogeneous(self) -> bool:
        """Whether any parameter is heterogeneous."""
        return any(p.is_heterogeneous for p in self.parameters)

    @property
    def is_homogeneous(self) -> bool:
        """Whether all parameters are homogeneous."""
        return not self.is_heterogeneous

    @property
    def data(self) -> tuple[jnp.ndarray, ...]:
        """Tuple of parameter data arrays."""
        return tuple(p.data for p in self.parameters)

    @property
    def aligned(self) -> Self:
        """Return a version of the parameters with aligned shapes."""
        if self.is_homogeneous:
            return self
        shape = self.shape
        params = tuple(
            p.replace(data=jnp.full(shape, p.data)) if p.is_scalar else p
            for p in self.parameters
        )
        return self.replace(params=params)

    @property
    def array(self) -> jnp.ndarray:
        """Column stacked array of parameter data."""
        return jnp.column_stack(self.aligned.data)

    @property
    def outer(self) -> tuple[LazyOuter, ...]:
        """Lazy outer sums of the parameter values."""
        return tuple(p.outer for p in self.parameters)

    @property
    def subset(self) -> "_ParametersIndexer":
        """Indexer for subsetting parameters."""
        return _ParametersIndexer(self)

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.names == other.names
            and all(
                p1.equals(p2)
                for p1, p2 in zip(self.parameters, other.parameters, strict=True)
            )
        )

    @abstractmethod
    def _check_names(self) -> None:
        """Check that parameter names are valid."""

    @classmethod
    def from_arrays(cls, *arrays: jnp.ndarray | None) -> Self:
        """Create parameters from arrays.

        Parameters
        ----------
        *arrays
            Arrays to create parameters from.
        """
        if len(arrays) == 1 and isinstance(arrays[0], cls):
            return cls.from_arrays(*arrays[0])
        if len(arrays) < len(cls.definition):
            arrays += (None,) * (len(cls.definition) - len(arrays))
        elif len(arrays) > len(cls.definition):
            errmsg = f"expected at most {len(cls.definition)} arrays, got {len(arrays)}"
            raise ValueError(errmsg)
        params = tuple(
            d(a) if not isinstance(a, d) else d() if a is None else a
            for a, d in zip(arrays, cls.definition, strict=True)
        )
        return cls(params)


class _ParametersIndexer(eqx.Module):
    parameters: AbstractParameters

    def __getitem__(self, args: Any) -> AbstractParameters:
        params = tuple(p if jnp.isscalar(p) else p[args] for p in self.parameters)
        return self.parameters.replace(parameters=params)
