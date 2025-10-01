from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg.statistics.abc import (
    AbstractErgmNodePairStatistic,
    AbstractErgmNodeStatistic,
    AbstractErgmViewStatistic,
    AbstractStatistic,
)

if TYPE_CHECKING:
    from grgg.models.abc import AbstractModel, AbstractModelModule
    from grgg.models.ergm.abc import (
        AbstractErgmNodePairView,
        AbstractErgmNodeView,
        AbstractErgmView,
    )
    from grgg.models.ergm.abc.motifs import (
        AbstractErgmMotifs,
        AbstractErgmNodeMotifs,
        AbstractErgmNodePairMotifs,
    )

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
    """Abstract base class for motif statistics."""

    motifs: eqx.AbstractVar[TT]


class AbstractErgmViewMotifStatistic[MT](
    AbstractErgmViewStatistic[MT], AbstractMotifStatistic[MT]
):
    """Abstract base class for motif statistics on model views."""

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
    """Abstract base class for node motif statistics."""

    @property
    def nodes(self) -> MV:
        """The node motifs of the view the statistic is computed on."""
        return self.module.nodes


class AbstractErgmNodePairMotifStatistic[ME](
    AbstractErgmNodePairStatistic[ME], AbstractErgmViewMotifStatistic[ME]
):
    """Abstract base class for node pair motif statistics."""

    @property
    def pairs(self) -> ME:
        """The node pair motifs of the view the statistic is computed on."""
        return self.module.pairs
