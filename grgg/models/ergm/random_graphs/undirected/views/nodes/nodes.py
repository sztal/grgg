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
    StructuralComplementarity,
    StructuralSimilarity,
    TClosure,
    TClustering,
)

from .complementarity import UndirectedRandomGraphStructuralComplementarity
from .degree import UndirectedRandomGraphDegreeStatistic
from .qclosure import UndirectedRandomGraphQClosure
from .qclust import UndirectedRandomGraphQClustering
from .similarity import UndirectedRandomGraphStructuralSimilarity
from .tclosure import UndirectedRandomGraphTClosure
from .tclust import UndirectedRandomGraphTClustering

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
