from abc import abstractmethod
from typing import Any

import jax.numpy as jnp

from grgg.abc import AbstractModule

__all__ = ("AbstractFunction",)


class AbstractFunction(AbstractModule):
    """Abstract base class for model functions."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
        """Function implementation."""

    def equals(self, other: Any) -> bool:
        return super().equals(other)
