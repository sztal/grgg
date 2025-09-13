from abc import ABC, abstractmethod
from typing import Any, Self

from flax import nnx

__all__ = ("AbstractComponent", "AbstractModule", "AbstractGRGG")


class AbstractComponent(ABC):
    """Abstract base class for model elements."""

    @abstractmethod
    def equals(self, other: object) -> bool:
        """Check equality with another model element."""
        t1 = type(self)
        t2 = type(other)
        return issubclass(t1, t2) or issubclass(t2, t1)

    @abstractmethod
    def __copy__(self) -> Self:
        """Create a copy of the model element."""

    def copy(self, **kwargs: Any) -> Self:
        """Create a copy of the model element."""
        return self.__copy__(**kwargs)


class AbstractModule(AbstractComponent, nnx.Module):
    """Abstract base class for modules."""


class AbstractGRGG(ABC):  # noqa
    """Abstract base class for GRGG models."""

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if any(getattr(c, "__grgg_model__", False) for c in subclass.__mro__):
            return True
        return NotImplemented
