from typing import TYPE_CHECKING, ClassVar, TypeVar

import jax.numpy as jnp

from grgg._typing import Reals

from .abc import AbstractErgmNodeStatistic

if TYPE_CHECKING:
    from grgg.models.base.ergm.model import AbstractErgm, E, P, S, V
    from grgg.models.base.model import AbstractModel, AbstractModelModule

    T = TypeVar("T", bound=AbstractModel)
    M = TypeVar("M", bound=AbstractModelModule[T])
    TE = TypeVar("TE", bound=AbstractErgm[P, V, E, S])
    ME = TypeVar("ME", bound=AbstractModelModule[TE])

__all__ = (
    "AbstractErgmNodeLocalStructureStatistic",
    "TClustering",
    "TClosure",
    "StructuralSimilarity",
    "TStatistics",
    "QClustering",
    "QClosure",
    "StructuralComplementarity",
    "QStatistics",
)


VT = TypeVar("VT", bound="V")


class AbstractErgmNodeLocalStructureStatistic[VT](AbstractErgmNodeStatistic[VT]):
    def postprocess(self, moment: Reals) -> Reals:
        moment = super().postprocess(moment)
        return jnp.clip(moment, 0, 1)


# Triangle-based statistics ----------------------------------------------------------


class TClustering(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class TClosure(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class StructuralSimilarity(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class TStatistics(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


# Quadrangle-based statistics --------------------------------------------------------


class QClustering(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class QClosure(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class StructuralComplementarity(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class QStatistics(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)
