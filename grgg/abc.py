from abc import ABC, abstractmethod
from typing import Self

from flax import nnx

__all__ = ("AbstractComponent", "AbstractModule")


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

    def copy(self) -> Self:
        """Create a copy of the model element."""
        return self.__copy__()


class AbstractModule(AbstractComponent, nnx.Module):
    """Abstract base class for modules."""
