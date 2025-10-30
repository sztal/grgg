from abc import abstractmethod

from .modules import AbstractModelModule

__all__ = ("AbstractModelFunctions",)


class AbstractModelFunctions[T](AbstractModelModule[T]):
    """Abstract base class for model functions."""

    model: T

    def __init__(self, model: T) -> None:
        self.model = model
        self.compile()

    @abstractmethod
    def compile(self) -> None:
        """Bind model functions to the model instance and compile."""
