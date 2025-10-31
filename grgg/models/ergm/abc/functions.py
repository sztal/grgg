from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from grgg._typing import Reals
from grgg.models.abc import AbstractModelFunctions

if TYPE_CHECKING:
    from .models import AbstractErgm

    T = TypeVar("T", bound="AbstractErgm")

__all__ = ("AbstractErgmFunctions",)


class AbstractErgmFunctions[T](AbstractModelFunctions[T]):
    """Abstract base class for ERGM functions."""

    @abstractmethod
    def compile_free_energy(self) -> Callable[..., Reals]:
        """Compile the free energy function."""

    @abstractmethod
    def compile_hamiltonian(self) -> Callable[..., Reals]:
        """Compile the Hamiltonian function."""
