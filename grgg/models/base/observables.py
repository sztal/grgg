from collections.abc import Mapping
from typing import Any, ClassVar

from grgg._typing import Numbers
from grgg.utils.variables import ArrayBundle, Constraints, Variable

__all__ = ("AbstractObservable", "AbstractObservables", "Degree")


class AbstractObservable(Variable):
    """Abstract base class for model observables.

    Attributes
    ----------
    data
        Observable value(s).
    """

    data: Numbers


class AbstractObservables(ArrayBundle[AbstractObservable]):
    """Container for model observables."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        for attr in ("statistic", "parameter"):
            for name in cls.get_instance_fields():
                field = cls.__dataclass_fields__[name]
                if attr not in field.metadata:
                    errmsg = (
                        f"'{cls.__name__}.{name}' is missing required "
                        f"'{attr}' metadata attribute"
                    )
                    raise TypeError(errmsg)

    @property
    def fields(self) -> Mapping[str, Any]:
        return self.get_fields()

    @classmethod
    def get_fields(cls) -> Mapping[str, Any]:
        return {
            k: v for k, v in cls.__dataclass_fields__.items() if k in cls.get_names()
        }


# Concrete observables ---------------------------------------------------------------


class Degree(AbstractObservable):
    """Node degree(s)."""

    constraints: ClassVar[Constraints] = Constraints("real", lower=0.0)
