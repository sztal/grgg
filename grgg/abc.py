from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import replace
from typing import Any, Self

import equinox as eqx

__all__ = ("AbstractModule", "AbstractGRGG")


class AbstractModule(eqx.Module):
    """Abstract base class for model elements."""

    @abstractmethod
    def equals(self, other: object) -> bool:
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


class AbstractGRGG(ABC):  # noqa
    """Abstract base class for GRGG models."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if any(getattr(c, "__grgg_model__", False) for c in subclass.__mro__):
            return True
        return NotImplemented
