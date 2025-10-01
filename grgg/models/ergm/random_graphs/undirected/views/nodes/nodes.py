from typing import TYPE_CHECKING, ClassVar, TypeVar

from grgg.models.ergm.random_graphs.undirected.abc import (
    AbstractUndirectedRandomGraphNodeView,
)
from grgg.models.ergm.random_graphs.undirected.motifs.nodes.motifs import (
    UndirectedRandomGraphNodeMotifs,
)
from grgg.statistics import (
    DegreeStatistic,
    QClosureStatistic,
    QClusteringStatistic,
    TClosureStatistic,
    TClusteringStatistic,
)

from .statistics import (
    UndirectedRandomGraphDegreeStatistic,
    UndirectedRandomGraphQClosureStatistic,
    UndirectedRandomGraphQClusteringStatistic,
    UndirectedRandomGraphTClosureStatistic,
    UndirectedRandomGraphTClusteringStatistic,
)

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


@DegreeStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphDegreeStatistic:
    return UndirectedRandomGraphDegreeStatistic(nodes)


@TClusteringStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphTClusteringStatistic:
    return UndirectedRandomGraphTClusteringStatistic(nodes)


@TClosureStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphTClosureStatistic:
    return UndirectedRandomGraphTClosureStatistic(nodes)


@QClusteringStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphQClusteringStatistic:
    return UndirectedRandomGraphQClusteringStatistic(nodes)


@QClosureStatistic.from_module.register
@classmethod
def _(
    cls,  # noqa
    nodes: UndirectedRandomGraphNodeView,  # noqa
) -> UndirectedRandomGraphQClosureStatistic:
    return UndirectedRandomGraphQClosureStatistic(nodes)
