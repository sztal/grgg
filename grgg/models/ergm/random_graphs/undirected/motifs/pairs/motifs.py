from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc import AbstractErgmNodePairMotifs

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import RandomGraph
    from grgg.models.ergm.random_graphs.undirected.views import (
        RandomGraphNodePairView,
        RandomGraphNodeView,
    )

    T = TypeVar("T", bound="RandomGraph")
    V = TypeVar("V", bound="RandomGraphNodeView")
    E = TypeVar("E", bound="RandomGraphNodePairView")


class RandomGraphNodePairMotifs[E](AbstractErgmNodePairMotifs[E]):
    """Class for undirected node pair motifs."""

    view: "RandomGraphNodePairView"
