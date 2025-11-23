from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from .modules import AbstractModelModule

if TYPE_CHECKING:
    from .model import AbstractModel

__all__ = ("AbstractModelOptimizer",)


T = TypeVar("T", bound="AbstractModel")


class AbstractModelOptimizer[T](AbstractModelModule[T]):
    """Abstract base class for model optimizers."""

    model: T

    def __call__(self, *args: Any, **kwargs: Any) -> T:
        return self.optimize(*args, **kwargs)

    @abstractmethod
    def optimize(self, *args: Any, **kwargs: Any) -> T:
        """Optimize the model."""

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.model.equals(other.model)
