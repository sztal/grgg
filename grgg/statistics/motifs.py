from typing import TYPE_CHECKING, ClassVar, TypeVar

import equinox as eqx

from grgg.statistics.abc import (
    AbstractErgmNodePairStatistic,
    AbstractErgmNodeStatistic,
    AbstractErgmViewStatistic,
    AbstractStatistic,
)

if TYPE_CHECKING:
    from grgg.models.base.ergm import (
        AbstractErgmNodePairView,
        AbstractErgmNodeView,
        AbstractErgmView,
    )
    from grgg.models.base.ergm.motifs import (
        AbstractErgmMotifs,
        AbstractErgmNodeMotifs,
        AbstractErgmNodePairMotifs,
    )
    from grgg.models.base.model import AbstractModel, AbstractModelModule

    T = TypeVar("T", bound=AbstractModel)
    TT = TypeVar("TT", bound=AbstractModelModule[T])
    VT = TypeVar("VT", bound=AbstractErgmView)
    VV = TypeVar("VV", bound=AbstractErgmNodeView)
    VE = TypeVar("VE", bound=AbstractErgmNodePairView)
    MT = TypeVar("MT", bound=AbstractErgmMotifs[VT])
    MV = TypeVar("MV", bound=AbstractErgmNodeMotifs[VV])
    ME = TypeVar("ME", bound=AbstractErgmNodePairMotifs[VE])

__all__ = ("AbstractErgmNodeMotifStatistic",)


TT = TypeVar("TT", bound="TT")
MT = TypeVar("MT", bound="MT")
MV = TypeVar("MV", bound="MV")
ME = TypeVar("ME", bound="ME")


class AbstractMotifStatistic[TT](AbstractStatistic[TT]):
    motifs: eqx.AbstractVar[TT]


class AbstractErgmViewMotifStatistic[MT](
    AbstractErgmViewStatistic[MT], AbstractMotifStatistic[MT]
):
    @property
    def motifs(self) -> MT:
        """The motifs of the view the statistic is computed on."""
        return self.module

    @property
    def view(self) -> MT:
        """The view the statistic is computed on."""
        return self.module.view


class AbstractErgmNodeMotifStatistic[MV](
    AbstractErgmNodeStatistic[MV], AbstractErgmViewMotifStatistic[MV]
):
    supported_moments: ClassVar[tuple[int, ...]] = (1,)

    @property
    def nodes(self) -> MV:
        """The node motifs of the view the statistic is computed on."""
        return self.module.nodes


class AbstractErgmNodePairMotifStatistic[ME](
    AbstractErgmNodePairStatistic[ME], AbstractErgmViewMotifStatistic[ME]
):
    supported_moments: ClassVar[tuple[int, ...]] = (1,)

    @property
    def pairs(self) -> ME:
        """The node pair motifs of the view the statistic is computed on."""
        return self.module.pairs


# Node motif statistics --------------------------------------------------------------


class TWedgeMotif(AbstractErgmNodeMotifStatistic):
    module: MV

    label: ClassVar[str] = "twedge"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class THeadMotif(AbstractErgmNodeMotifStatistic):
    module: MV

    label: ClassVar[str] = "thead"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class TriangleMotif(AbstractErgmNodeMotifStatistic):
    module: MV

    label: ClassVar[str] = "triangle"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class QWedgeMotif(AbstractErgmNodeMotifStatistic):
    module: MV

    label: ClassVar[str] = "qwedge"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)


class QHeadMotif(AbstractErgmNodeMotifStatistic):
    module: MV

    label: ClassVar[str] = "qhead"


class QuadrangleMotif(AbstractErgmNodeMotifStatistic):
    module: MV

    label: ClassVar[str] = "quadrangle"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)
