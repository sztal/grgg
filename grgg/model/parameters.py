from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Scalar, Vector

ParamT = Scalar | Vector


class CONSTRAINTS(eqx.Enumeration):
    """Enumeration of supported parameter constraints."""

    real = "real"
    non_negative = "non-negative"


class AbstractParameterSpecification(ABC):
    """Abstract base class for parameter specifications."""

    def __call__(self, value: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.validate(value)

    @property
    @abstractmethod
    def name(self) -> str:
        """Parameter name."""

    @property
    @abstractmethod
    def constraints(self) -> list[eqx._enum.EnumerationItem, ...]:
        """List of constraints."""
        return []

    @property
    @abstractmethod
    def default_value(self) -> ParamT:
        """Default parameter value."""

    @classmethod
    def validate(cls, value: jnp.ndarray | None) -> None:
        """Check all constraints."""
        spec = cls()
        if value is None:
            value = spec.default_value
        value = cls._validate(value)
        for constraint in spec.constraints:
            spec._check_constraint(value, constraint)
        return value

    @classmethod
    @abstractmethod
    def _validate(cls, value: jnp.ndarray) -> jnp.ndarray:
        """Validate input value."""
        value = jnp.asarray(value)
        if value.size <= 0:
            errmsg = "parameter value must be a non-empty array"
            raise ValueError(errmsg)
        return value

    def _check_constraint(
        self, value: jnp.ndarray, constraint: eqx._enum.EnumerationItem
    ) -> None:
        """Check constraint."""
        if constraint == CONSTRAINTS.real:
            self._error_if(constraint, ~jnp.isreal(value).all())
        elif constraint == CONSTRAINTS.non_negative:
            self._error_if(constraint, (value < 0).any())
        else:
            errmsg = f"unknown constraint: {constraint}"
            raise ValueError(errmsg)

    def _error_if(self, constraint, condition: bool) -> None:
        name = constraint._enumeration[constraint]
        if condition:
            errmsg = f"'{self.name}' must be {name}"
            raise ValueError(errmsg)


class AbstractNodeParameterSpecification(AbstractParameterSpecification):
    """Abstract base class for node parameter specifications."""

    @property
    def constraints(self) -> list[eqx._enum.EnumerationItem, ...]:
        return [*super().constraints, CONSTRAINTS.real]

    @classmethod
    def _validate(cls, value: jnp.ndarray) -> jnp.ndarray:
        value = super()._validate(value)
        if value.ndim > 1:
            errmsg = "node parameter value must be a scalar or a 1D array"
            raise ValueError(errmsg)
        return value

    @classmethod
    def validate(cls, value: jnp.ndarray) -> jnp.ndarray:
        """Check all constraints."""
        return super().validate(value).astype(float)


class Beta(AbstractNodeParameterSpecification):
    """Beta parameter (inverse temperature).

    It controls the strength of the coupling between the network topology
    and the underlying geometry.

    Attributes
    ----------
    value
        Parameter value. Default is 1.5.

    Examples
    --------
    >>> beta = Beta()
    >>> beta()  # default value
    Array(1.5, ...)
    >>> beta(2.0)  # homogeneous value
    Array(2.0, ...)
    >>> beta([1,2,3])  # heterogeneous value
    Array([1., 2., 3.], ...)

    Error is raised for invalid values.
    >>> beta(-1)  # negative value
    Traceback (most recent call last):
        ...
    ValueError: 'beta' must be non-negative
    """

    @property
    def name(self) -> str:
        return "beta"

    @property
    def constraints(self) -> list[eqx._enum.EnumerationItem, ...]:
        return [*super().constraints, CONSTRAINTS.non_negative]

    @property
    def default_value(self) -> ParamT:
        return jnp.array(1.5)


class Mu(AbstractNodeParameterSpecification):
    """Mu parameter (chemical potential).

    It controls the average degree of the network.

    Attributes
    ----------
    value
        Parameter value. Default is 0.0.

    Examples
    --------
    >>> mu = Mu()
    >>> mu()  # default value
    Array(0.0, ...)
    >>> mu(1.0)  # homogeneous value
    Array(1.0, ...)
    >>> mu([1, 2, 3])  # heterogeneous value
    Array([1., 2., 3.], ...)

    Error is raised for invalid values.
    >>> mu(1+1j)  # negative value
    Traceback (most recent call last):
        ...
    ValueError: 'mu' must be real
    """

    @property
    def name(self) -> str:
        return "mu"

    @property
    def default_value(self) -> ParamT:
        return jnp.array(0.0)
