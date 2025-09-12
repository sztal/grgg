from abc import abstractmethod
from typing import Any, Self

import jax.numpy as np
from flax import nnx

from grgg._typing import Floats, Scalar, Vector
from grgg.abc import AbstractComponent
from grgg.lazy import LazyOuter

ParamT = Scalar | Vector


class AbstractParameterConstraint(AbstractComponent):
    """Abstract base class for parameter constraints."""

    @abstractmethod
    def validate(self, param: "AbstractModelParameter") -> None:
        """Validate the parameter value."""

    def __copy__(self) -> Self:
        return type(self)()

    def equals(self, other: object) -> bool:
        return super().equals(other)


class NonNegative(AbstractParameterConstraint):
    """Constraint that ensures a parameter is non-negative."""

    def validate(
        self, param: "AbstractModelParameter", name: str | None = None
    ) -> None:
        if np.any(param.value < 0):
            if not name:
                name = type(param).__name__
            errmsg = f"parameter '{name}' must be non-negative"
            raise ValueError(errmsg)


class AbstractModelVariable(AbstractComponent, nnx.Variable[Floats]):
    """Abstract base class for model variables."""

    constraints: tuple[AbstractParameterConstraint, ...] = ()

    def __init__(self, value: Floats, **kwargs: Any) -> None:
        value = self.validate_value(value)
        super().__init__(value, **kwargs)

    def __copy__(self) -> Self:
        return super().__copy__()

    def equals(self, other: object) -> bool:
        return super().equals(other) and np.array_equal(self.value, other.value).item()

    @classmethod
    @abstractmethod
    def validate_value(cls, value: Floats) -> Floats:
        """Validate the variable value."""
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.integer):
            value = value.astype(float)
        if value.size <= 0:
            errmsg = "Value must be a non-empty array"
            raise ValueError(errmsg)
        if not np.issubdtype(value.dtype, np.floating):
            errmsg = f"Value must be a float array, got '{value.dtype}'"
            raise TypeError(errmsg)
        if value.size == 1:
            value = value.flatten()[0]  # Convert single-element array to scalar
        for constraint in cls.constraints:
            constraint.validate(value, cls.__name__)
        return value


class AbstractModelParameter(AbstractModelVariable, nnx.Param[ParamT]):
    """Abstract base class for model parameters."""

    def __init__(self, value: ParamT | None = None, **kwargs: Any) -> None:
        if value is None:
            value = self.default_value
        value = self.validate_value(value)
        super().__init__(value, **kwargs)

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the parameter is heterogeneous (varies across nodes)."""
        return self.value.size > 1

    @property
    def outer(self) -> LazyOuter:
        """Lazy outer sum of the parameter with itself."""
        value = self.value if self.is_heterogeneous else self.value / 2
        return LazyOuter(value, value, op=np.add)

    @classmethod
    def validate_value(cls, value: ParamT) -> ParamT:
        """Validate the parameter value."""
        value = super().validate_value(value)
        if value.ndim > 1:
            errmsg = "Parameter value must be a scalar or 1D array"
            raise ValueError(errmsg)
        return value

    @property
    @abstractmethod
    def default_value(self) -> ParamT:
        """Default parameter value."""


class Beta(AbstractModelParameter):
    """Beta parameter (inverse temperature).

    It controls the strength of the coupling between the network topology
    and the underlying geometry.

    Attributes
    ----------
    value
        Parameter value. Default is 1.5.

    Examples
    --------
    >>> beta = Beta()  # default value
    >>> beta.value
    Array(1.5, ...)
    >>> beta = Beta(2.0)  # homogeneous value
    >>> beta.value
    Array(2.0, ...)
    >>> beta.is_heterogeneous
    False
    >>> beta = Beta([1,2,3])  # heterogeneous value
    >>> beta.value
    Array([1., 2., 3.], ...)
    >>> beta.is_heterogeneous
    True

    Equalit checks are implemented via the `equals` method
    as not to interfere with Flax's internal mechanisms.
    >>> Beta().equals(Beta())
    True
    >>> Beta([1,2]).equals(Beta([1,2]))
    True
    """

    @property
    def default_value(self) -> ParamT:
        return np.array(1.5)  # Default beta value

    def validate_value(self, value: ParamT) -> ParamT:
        value = super().validate_value(value)
        if np.any(value < 0):
            errmsg = "beta must be non-negative"
            raise ValueError(errmsg)
        return value


class Mu(AbstractModelParameter):
    """Mu parameter (chemical potential).

    It controls the average degree of the network.

    Attributes
    ----------
    value
        Parameter value. Default is 0.0.

    Examples
    --------
    >>> mu = Mu()  # default value
    >>> mu.value
    Array(0.0, ...)
    >>> mu.is_heterogeneous
    False
    >>> mu = Mu([1, 2, 3])  # heterogeneous value
    >>> mu.value
    Array([1., 2., 3.], ...)
    >>> mu.is_heterogeneous
    True

    Equalit checks are implemented via the `equals` method
    as not to interfere with Flax's internal mechanisms.
    >>> Mu().equals(Mu())
    True
    >>> Mu([1,2]).equals(Mu([1,2]))
    True
    """

    @property
    def default_value(self) -> ParamT:
        return np.array(0.0)  # Default mu value

    def validate_value(self, value: ParamT) -> ParamT:
        value = super().validate_value(value)
        return value
