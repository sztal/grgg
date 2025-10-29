from typing import TYPE_CHECKING, ClassVar, TypeVar

from grgg.models.ergm.abc import AbstractErgmNodeMotifs

from ._qhead import RandomGraphQHeadMotif
from ._quadrangle import RandomGraphQuadrangleMotif
from ._qwedge import RandomGraphQWedgeMotif
from ._thead import RandomGraphTHeadMotif
from ._triangle import RandomGraphTriangleMotif
from ._twedge import RandomGraphTWedgeMotif

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import RandomGraph
    from grgg.models.ergm.random_graphs.undirected.views import (
        RandomGraphNodePairView,
        RandomGraphNodeView,
    )

    T = TypeVar("T", bound="RandomGraph")
    V = TypeVar("V", bound="RandomGraphNodeView")
    E = TypeVar("E", bound="RandomGraphNodePairView")


class RandomGraphNodeMotifs[V, T](AbstractErgmNodeMotifs[V, T]):
    """Class for undirected node motifs."""

    view: "RandomGraphNodeView"

    triangle_cls: ClassVar[type[RandomGraphTriangleMotif]] = RandomGraphTriangleMotif
    twedge_cls: ClassVar[type[RandomGraphTWedgeMotif]] = RandomGraphTWedgeMotif
    thead_cls: ClassVar[type[RandomGraphTHeadMotif]] = RandomGraphTHeadMotif
    quadrangle_cls: ClassVar[
        type[RandomGraphQuadrangleMotif]
    ] = RandomGraphQuadrangleMotif
    qwedge_cls: ClassVar[type[RandomGraphQWedgeMotif]] = RandomGraphQWedgeMotif
    qhead_cls: ClassVar[type[RandomGraphQHeadMotif]] = RandomGraphQHeadMotif
