from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Self

import jax.numpy as np
from flax import nnx

from ._pairs import NodePairs
from ._typing import Floats

__all__ = ("AbstractComponent", "AbstractModelModule")


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


class AbstractModelModule(AbstractModule):
    """Abstract base class for model modules."""

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes in the model."""

    @property
    def pairs(self) -> NodePairs:
        """Node pairs indexer."""
        return NodePairs(self)

    @abstractmethod
    def define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
        """Define the module function."""

    def _define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
        function = self.define_function()

        @wraps(function)
        def wrapper(*args: Floats) -> Floats:
            args = tuple(np.asarray(a) for a in args)
            return function(*args)

        return nnx.jit(wrapper)
