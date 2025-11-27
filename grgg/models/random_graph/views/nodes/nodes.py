from typing import TYPE_CHECKING, Literal, TypeVar

from grgg.models.base.ergm import ErgmNodeMotifs
from grgg.models.base.random_graphs import AbstractRandomGraphNodeView

from ._complementarity import RandomGraphStructuralComplementarity
from ._degree import RandomGraphDegree
from ._qclosure import RandomGraphQClosure
from ._qclust import RandomGraphQClustering
from ._qstats import RandomGraphQStatistics
from ._similarity import RandomGraphStructuralSimilarity
from ._tclosure import RandomGraphTClosure
from ._tclust import RandomGraphTClustering
from ._tstats import RandomGraphTStatistics
from .motifs._qhead import RandomGraphQHeadMotif
from .motifs._quadrangle import RandomGraphQuadrangleMotif
from .motifs._qwedge import RandomGraphQWedgeMotif
from .motifs._thead import RandomGraphTHeadMotif
from .motifs._triangle import RandomGraphTriangleMotif
from .motifs._twedge import RandomGraphTWedgeMotif

if TYPE_CHECKING:
    from ...model import RandomGraph

__all__ = ("RandomGraphNodeView",)


T = TypeVar("T", bound="RandomGraph")


class RandomGraphNodeView[T](AbstractRandomGraphNodeView[T]):
    """Nodes view for undirected random graph models."""

    # Register statistics methods ----------------------------------------------------

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["degree"]) -> RandomGraphDegree:
        return RandomGraphDegree(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["tclust"]) -> RandomGraphTClustering:
        return RandomGraphTClustering(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["tclosure"]) -> RandomGraphTClosure:
        return RandomGraphTClosure(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["similarity"]) -> RandomGraphStructuralSimilarity:
        return RandomGraphStructuralSimilarity(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["qclust"]) -> RandomGraphQClustering:
        return RandomGraphQClustering(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["qclosure"]) -> RandomGraphQClosure:
        return RandomGraphQClosure(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["complementarity"]) -> RandomGraphStructuralComplementarity:
        return RandomGraphStructuralComplementarity(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["tstats"]) -> RandomGraphTStatistics:
        return RandomGraphTStatistics(self)

    @AbstractRandomGraphNodeView._get_statistic.dispatch
    def _(self, _: Literal["qstats"]) -> RandomGraphQStatistics:
        return RandomGraphQStatistics(self)

    # Register motif counts methods --------------------------------------------------

    @ErgmNodeMotifs._get_motif.dispatch
    def _(
        self, _: Literal["twedge"], __: "RandomGraphNodeView"
    ) -> RandomGraphTWedgeMotif:
        return RandomGraphTWedgeMotif(self)

    @ErgmNodeMotifs._get_motif.dispatch
    def _(
        self, _: Literal["thead"], __: "RandomGraphNodeView"
    ) -> RandomGraphTHeadMotif:
        return RandomGraphTHeadMotif(self)

    @ErgmNodeMotifs._get_motif.dispatch
    def _(
        self, _: Literal["qwedge"], __: "RandomGraphNodeView"
    ) -> RandomGraphQWedgeMotif:
        return RandomGraphQWedgeMotif(self)

    @ErgmNodeMotifs._get_motif.dispatch
    def _(
        self, _: Literal["qhead"], __: "RandomGraphNodeView"
    ) -> RandomGraphQHeadMotif:
        return RandomGraphQHeadMotif(self)

    @ErgmNodeMotifs._get_motif.dispatch
    def _(
        self, _: Literal["triangle"], __: "RandomGraphNodeView"
    ) -> RandomGraphTriangleMotif:
        return RandomGraphTriangleMotif(self)

    @ErgmNodeMotifs._get_motif.dispatch
    def _(
        self, _: Literal["quadrangle"], __: "RandomGraphNodeView"
    ) -> RandomGraphQuadrangleMotif:
        return RandomGraphQuadrangleMotif(self)
