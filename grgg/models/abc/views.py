from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

from grgg.utils.indexing import IndexArgT

from .modules import AbstractModelModule

if TYPE_CHECKING:
    pass

__all__ = ("AbstractModelView",)


class AbstractModelView[T](AbstractModelModule[T]):
    """Abstract base class for model views."""

    model: T

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the view is active"""

    @property
    @abstractmethod
    def _default_index_args(self) -> IndexArgT | tuple[IndexArgT, ...]:
        """Default index for the view."""
        return ()

    @abstractmethod
    def __getitem__(self, args: IndexArgT | tuple[IndexArgT, ...]) -> Self:
        """Indexing method."""

    @abstractmethod
    def reset(self) -> Self:
        """Reset the view to its default state."""

    @property
    def parameters(self) -> "T.Parameters":
        """Tuple of parameter arrays for the view."""
        return self.model.Parameters(
            *(self.get_parameter(name) for name in self.model.parameters._fields)
        )

    @abstractmethod
    def get_parameter(self, args: Any, **kwargs: Any) -> Any:
        """Get a model parameter by index or name."""

    @abstractmethod
    def _equals(self, other: object) -> bool:
        """Check equality with another view."""
        return super()._equals(other) and self.model.equals(other.model)

    @abstractmethod
    def materialize(self, *, copy: bool = False) -> T:
        """Materialize a new model with the current selection."""
