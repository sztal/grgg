from typing import TYPE_CHECKING, ClassVar, TypeVar

import jax.numpy as jnp

from grgg._typing import Reals

from .abc import AbstractErgmNodeStatistic

if TYPE_CHECKING:
    from grgg.models.abc import AbstractModel, AbstractModelModule
    from grgg.models.ergm.abc.models import AbstractErgm, E, P, S, V

    T = TypeVar("T", bound=AbstractModel)
    M = TypeVar("M", bound=AbstractModelModule[T])
    TE = TypeVar("TE", bound=AbstractErgm[P, V, E, S])
    ME = TypeVar("ME", bound=AbstractModelModule[TE])

VT = TypeVar("VT", bound="V")


class AbstractErgmNodeLocalStructureStatistic[VT](AbstractErgmNodeStatistic[VT]):
    def postprocess(self, moment: Reals) -> Reals:
        moment = super().postprocess(moment)
        return jnp.clip(moment, 0, 1)


# Triangle-based statistics ----------------------------------------------------------


class TClustering(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "tclust"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class TClosure(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "tclosure"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class StructuralSimilarity(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "similarity"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class TStatistics(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "tstats"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


# Quadrangle-based statistics --------------------------------------------------------


class QClustering(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "qclust"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class QClosure(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "qclosure"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class StructuralComplementarity(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "complementarity"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class QStatistics(AbstractErgmNodeLocalStructureStatistic):
    module: VT

    label: ClassVar[str] = "qstats"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)
