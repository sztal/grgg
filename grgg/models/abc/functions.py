from typing import TYPE_CHECKING, TypeVar

from .modules import AbstractModelModule

if TYPE_CHECKING:
    from .model import AbstractModel

    T = TypeVar("T", bound=AbstractModel)

__all__ = ("AbstractModelFunctions",)


class AbstractModelFunctions[T](AbstractModelModule[T]):
    """Abstract base class for model functions container.

    Attributes
    ----------
    model
        The model class associated with the functions.
    """

    model: T

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.model.equals(other.model)
