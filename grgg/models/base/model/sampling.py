from typing import Any, TypeVar

from grgg.utils.dispatch import dispatch

from .modules import AbstractModelModule

__all__ = ("ModelSampler",)


T = TypeVar("T", bound="AbstractModel")


class ModelSampler[T](AbstractModelModule[T]):
    """Abstract base class for samplers."""

    model: T

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.sample(*args, **kwargs)

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.model.equals(other.model)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""
        return self._sample(self.model, *args, **kwargs)

    @dispatch.abstract
    def _sample(self, model: "AbstractModel", *args: Any, **kwargs: Any) -> Any:
        """Sample from the model."""


# Avoid circular import --------------------------------------------------------------

from .model import AbstractModel  # noqa
