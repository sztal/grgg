from abc import abstractmethod
from typing import ClassVar, Self

import equinox as eqx
import jax.numpy as jnp

from grgg.abc import AbstractModule

__all__ = (
    "Constraint",
    "Real",
    "Positive",
    "Negative",
    "NonPositive",
    "NonNegative",
)


class Constraint(AbstractModule):
    """Parameter constraint."""

    name: eqx.AbstractClassVar[str]
    available: ClassVar[dict[str, Self]] = {}

    def __init_subclass__(cls) -> None:
        if cls.name in cls.available:
            errmsg = f"constraint '{cls.name}' already registered"
            raise ValueError(errmsg)
        cls.available[cls.name] = cls()

    def __new__(cls) -> Self:
        return cls.available[cls.name]

    @abstractmethod
    def holds(self, data: jnp.ndarray) -> bool:
        """Check if data satisfies the constraint."""

    def check(self, data: jnp.ndarray, name: str | None = None) -> None:
        """Validate data against the constraint."""
        if not self.holds(data):
            errmsg = f"parameter is not '{self.name}'"
            if name:
                errmsg = f"'{name}' {errmsg}"
            raise ValueError(errmsg)


class Real(Constraint):
    """Real-valued constraint."""

    name = "real"

    def holds(self, data: jnp.ndarray) -> bool:
        return jnp.isreal(data).all()


class Positive(Constraint):
    """Positive-valued constraint."""

    name = "positive"

    def holds(self, data: jnp.ndarray) -> bool:
        return jnp.all(data > 0)


class Negative(Constraint):
    """Negative-valued constraint."""

    name = "negative"

    def holds(self, data: jnp.ndarray) -> bool:
        return jnp.all(data < 0)


class NonPositive(Constraint):
    """Non-positive-valued constraint."""

    name = "non-positive"

    def holds(self, data: jnp.ndarray) -> bool:
        return jnp.all(data <= 0)


class NonNegative(Constraint):
    """Non-negative-valued constraint."""

    name = "non-negative"

    def holds(self, data: jnp.ndarray) -> bool:
        return jnp.all(data >= 0)
