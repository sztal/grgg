from abc import abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import equinox as eqx

from .modules import AbstractModelModule

if TYPE_CHECKING:
    from .models import AbstractModel

__all__ = ("AbstractModelSampler",)

T = TypeVar("T", bound="AbstractModel")


class AbstractModelSampler[T](AbstractModelModule[T]):
    """Abstract base class for samplers."""

    model: eqx.AbstractVar[T]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.sample(*args, **kwargs)

    @abstractmethod
    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.model.equals(other.model)
