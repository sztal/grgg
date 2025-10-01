from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self, TypeVar

import jax.numpy as jnp

from grgg.utils.indexing import IndexArg

from .modules import AbstractModelModule

if TYPE_CHECKING:
    from .models import AbstractModel

T = TypeVar("T", bound="AbstractModel")

__all__ = ("AbstractModelView",)


class AbstractModelView[T](AbstractModelModule[T]):
    """Abstract base class for model views."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the view is active"""

    @property
    @abstractmethod
    def _default_index(self) -> IndexArg | tuple[IndexArg, ...]:
        """Default index for the view."""

    @abstractmethod
    def __getitem__(self, args: IndexArg | tuple[IndexArg, ...]) -> Self:
        """Indexing method."""

    @abstractmethod
    def reset(self) -> Self:
        """Reset the view to its default state."""

    @property
    def parameters(self) -> tuple[jnp.ndarray, ...]:
        """Tuple of parameter arrays for the view."""
        return tuple(self.get_parameter(name) for name in self.model.parameters.names)

    @abstractmethod
    def get_parameter(self, args: Any, **kwargs: Any) -> Any:
        """Get a model parameter by index or name."""

    @abstractmethod
    def equals(self, other: object) -> bool:
        """Check equality with another view."""
        return super().equals(other) and self.model.equals(other.model)

    @abstractmethod
    def materialize(self, *, copy: bool = False) -> T:
        """Materialize a new model with the current selection."""
