from functools import singledispatchmethod
from typing import Any, ClassVar, Self

from .abc import MV, AbstractErgmNodeMotifStatistic


class QHeadMotifStatistic(AbstractErgmNodeMotifStatistic):
    """Quadrangle head path motif statistic.

    Attributes
    ----------
    model
        The model statistics is computed for.
    label
        The label of the statistic.
    """

    module: MV

    label: ClassVar[str] = "qhead"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)

    @singledispatchmethod
    @classmethod
    def from_module(cls, module: object, *args: Any, **kwargs: Any) -> Self:  # noqa
        raise cls.unsupported_module_exception(module)
