from functools import singledispatchmethod
from typing import Any, ClassVar, Self

from .abc import VT, AbstractErgmNodeLocalStructureStatistic


class QClosure(AbstractErgmNodeLocalStructureStatistic):
    """Quadrangle closure statistic.

    Attributes
    ----------
    model
        The model statistics is computed for.
    label
        The label of the statistic.
    """

    module: VT

    label: ClassVar[str] = "qclosure"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)

    @singledispatchmethod
    @classmethod
    def from_module(cls, module: object, *args: Any, **kwargs: Any) -> Self:  # noqa
        raise cls.unsupported_module_exception(module)
