from typing import TYPE_CHECKING, ClassVar, TypeVar

from grgg.models.ergm.random_graphs.abc import (
    AbstractRandomGraphNodeView,
)

from ...motifs.nodes import RandomGraphNodeMotifs
from ...sampling import RandomGraphSampler
from ._complementarity import RandomGraphStructuralComplementarity
from ._degree import RandomGraphDegreeStatistic
from ._qclosure import RandomGraphQClosure
from ._qclust import RandomGraphQClustering
from ._qstats import RandomGraphQStatistics
from ._similarity import RandomGraphStructuralSimilarity
from ._tclosure import RandomGraphTClosure
from ._tclust import RandomGraphTClustering
from ._tstats import RandomGraphTStatistics

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import RandomGraph

__all__ = ("RandomGraphNodeView",)


T = TypeVar("T", bound="RandomGraph")
MV = TypeVar("MV", bound="RandomGraphNodeMotifs")
S = TypeVar("S", bound="RandomGraphSampler")


class RandomGraphNodeView[T, MV, S](AbstractRandomGraphNodeView[T, MV, S]):
    """Nodes view for undirected random graph models."""

    motifs_cls: ClassVar[type[MV]] = RandomGraphNodeMotifs  # type: ignore
    sampler_cls: ClassVar[type[S]] = RandomGraphSampler  # type: ignore

    degree_cls: ClassVar[type[RandomGraphDegreeStatistic]] = RandomGraphDegreeStatistic
    tclust_cls: ClassVar[type[RandomGraphTClustering]] = RandomGraphTClustering
    tclosure_cls: ClassVar[type[RandomGraphTClosure]] = RandomGraphTClosure
    similarity_cls: ClassVar[
        type[RandomGraphStructuralSimilarity]
    ] = RandomGraphStructuralSimilarity
    qclust_cls: ClassVar[type[RandomGraphQClustering]] = RandomGraphQClustering
    qclosure_cls: ClassVar[type[RandomGraphQClosure]] = RandomGraphQClosure
    complementarity_cls: ClassVar[
        type[RandomGraphStructuralComplementarity]
    ] = RandomGraphStructuralComplementarity
    tstats_cls: ClassVar[type[RandomGraphTStatistics]] = RandomGraphTStatistics
    qstats_cls: ClassVar[type[RandomGraphQStatistics]] = RandomGraphQStatistics
