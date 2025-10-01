from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc import AbstractErgmNodePairMotifs

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import UndirectedRandomGraph
    from grgg.models.ergm.random_graphs.undirected.views import (
        UndirectedRandomGraphNodePairView,
        UndirectedRandomGraphNodeView,
    )

    T = TypeVar("T", bound="UndirectedRandomGraph")
    V = TypeVar("V", bound="UndirectedRandomGraphNodeView")
    E = TypeVar("E", bound="UndirectedRandomGraphNodePairView")


class UndirectedRandomGraphNodePairMotifs[E](AbstractErgmNodePairMotifs[E]):
    """Class for undirected node pair motifs."""

    view: "UndirectedRandomGraphNodePairView"
