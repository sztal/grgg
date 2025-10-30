from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg._typing import Reals
from grgg.models.abc import AbstractModelFunctions

if TYPE_CHECKING:
    from .models import AbstractRandomGraph

T = TypeVar("T", bound="AbstractRandomGraph")

__all__ = ("AbstractRandomGraphFunctions",)

CouplingT = Callable[[Reals], Reals]


class AbstractRandomGraphFunctions[T](AbstractModelFunctions[T]):
    """Abstract base class for random graph model functions."""

    coupling: CouplingT = eqx.field(static=True)

    def compile(self) -> None:
        """Bind model functions to the model instance and compile."""
        self.coupling = self.compile_coupling()

    @abstractmethod
    def compile_coupling(self) -> CouplingT:
        """Compile the coupling function."""
