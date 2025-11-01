from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Reals
from grgg.models.abc import AbstractModelFunctions

if TYPE_CHECKING:
    from .model import AbstractErgm

    T = TypeVar("T", bound=AbstractErgm)

__all__ = ("AbstractErgmFunctions",)


class AbstractErgmFunctions[T](AbstractModelFunctions[T]):
    """Abstract base class for ERGM model functions."""

    @abstractmethod
    def free_energy(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the free energy of the model."""

    def partition_function(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute the partition function of the model."""
        return partition_function(self, *args, **kwargs)


@eqx.filter_jit
def partition_function(
    funcs: AbstractErgmFunctions, *args: Any, **kwargs: Any
) -> Reals:
    """Compute the partition function of the model."""
    free_energy = funcs.free_energy(*args, **kwargs)
    return jnp.exp(-free_energy)
