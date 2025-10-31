from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from grgg.models.abc import AbstractModelModule

if TYPE_CHECKING:
    from .models import AbstractModel

__all__ = ("AbstractModelFunctions",)


class AbstractModelFunctions[T](AbstractModelModule[T]):
    """Abstract base class for model functions."""

    names: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for name, attr in cls.__dict__.items():
            if name.startswith("compile_") and isinstance(attr, Callable):
                cls.names = cls.names.union({name})

    def __init__(self, model: "AbstractModel") -> None:
        self.model = model
        for name in self.names:
            fname = name.removeprefix("compile_")
            compiled = getattr(self, name)()
            object.__setattr__(self, fname, compiled)
