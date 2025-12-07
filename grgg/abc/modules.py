from abc import abstractmethod
from copy import deepcopy
from dataclasses import replace
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg.utils.misc import get_class_fields, get_instance_fields

__all__ = ("AbstractModule", "AbstractCallable", "AbstractFunction")


class AbstractModule(eqx.Module):
    """Abstract base class for general modules."""

    def equals(self, other: object) -> bool:
        """Check equality with another model element."""
        result = self._equals(other)
        if isinstance(result, jnp.ndarray):
            result = result.item()
        return bool(result)

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """Check equality with another model element."""
        t1 = type(self)
        t2 = type(other)
        return issubclass(t1, t2) or issubclass(t2, t1)

    def __copy__(self) -> Self:
        """Create a shallow copy of the model element."""
        return self.replace()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Create a deep copy."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            copied = deepcopy(v, memo)
            setattr(result, k, copied)
        return result

    def copy(self, *, deep: bool = False) -> Self:
        """Create a shallow or deep copy."""
        if deep:
            return self.__deepcopy__({})
        return self.__copy__()

    def replace(self, **kwargs: Any) -> Self:
        """Create a copy with some attributes replaced."""
        return replace(self, **kwargs)

    @classmethod
    def get_class_fields(cls) -> list[str]:
        """Get the list of class fields."""
        return get_class_fields(cls)

    @classmethod
    def get_instance_fields(cls) -> list[str]:
        """Get the list of instance fields."""
        return get_instance_fields(cls)


class AbstractCallable(AbstractModule):
    """Abstract base class for callable model elements."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Callable implementation."""


class AbstractFunction(AbstractCallable):
    """Abstract base class for model functions."""

    def _equals(self, other: Any) -> bool:
        return super()._equals(other)
