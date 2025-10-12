from typing import TYPE_CHECKING, ClassVar, TypeVar

from grgg.models.ergm.random_graphs.undirected.abc import (
    AbstractUndirectedRandomGraphNodeView,
)
from grgg.models.ergm.random_graphs.undirected.motifs.nodes.motifs import (
    UndirectedRandomGraphNodeMotifs,
)
from grgg.statistics import (
    Degree,
    QClosure,
    QClustering,
    QStatistics,
    StructuralComplementarity,
    StructuralSimilarity,
    TClosure,
    TClustering,
    TStatistics,
)

from ._complementarity import UndirectedRandomGraphStructuralComplementarity
from ._degree import UndirectedRandomGraphDegreeStatistic
from ._qclosure import UndirectedRandomGraphQClosure
from ._qclust import UndirectedRandomGraphQClustering
from ._qstats import UndirectedRandomGraphQStatistics
from ._similarity import UndirectedRandomGraphStructuralSimilarity
from ._tclosure import UndirectedRandomGraphTClosure
from ._tclust import UndirectedRandomGraphTClustering
from ._tstats import UndirectedRandomGraphTStatistics

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import UndirectedRandomGraph

__all__ = ("UndirectedRandomGraphNodeView",)


T = TypeVar("T", bound="UndirectedRandomGraph")
MV = TypeVar("MV", bound="UndirectedRandomGraphNodeMotifs")


class UndirectedRandomGraphNodeView[T, MV](
    AbstractUndirectedRandomGraphNodeView[T, MV]
):
    """Node view for undirected random graph models."""

    model: "UndirectedRandomGraph"
    motifs_cls: ClassVar[
        type[UndirectedRandomGraphNodeMotifs]
    ] = UndirectedRandomGraphNodeMotifs


# Register statistics implementations ------------------------------------------------


@Degree.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphDegreeStatistic:
    return UndirectedRandomGraphDegreeStatistic(nodes)


@TClustering.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphTClustering:
    return UndirectedRandomGraphTClustering(nodes)


@TClosure.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphTClosure:
    return UndirectedRandomGraphTClosure(nodes)


@StructuralSimilarity.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphStructuralSimilarity:
    return UndirectedRandomGraphStructuralSimilarity(nodes)


@TStatistics.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphTStatistics:
    return UndirectedRandomGraphTStatistics(nodes)


@QClustering.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphQClustering:
    return UndirectedRandomGraphQClustering(nodes)


@QClosure.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphQClosure:
    return UndirectedRandomGraphQClosure(nodes)


@StructuralComplementarity.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphStructuralComplementarity:
    return UndirectedRandomGraphStructuralComplementarity(nodes)


@QStatistics.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphQStatistics:
    return UndirectedRandomGraphQStatistics(nodes)
