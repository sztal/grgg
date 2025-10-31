from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc import AbstractErgmNodePairMotifs

if TYPE_CHECKING:
    from grgg.models.ergm.random_graphs.undirected.model import RandomGraph
    from grgg.models.ergm.random_graphs.undirected.views import (
        RandomGraphNodePairView,
    )

    T = TypeVar("T", bound="RandomGraph")


class RandomGraphNodePairMotifs[T](AbstractErgmNodePairMotifs[T]):
    """Class for undirected node pair motifs."""

    view: "RandomGraphNodePairView"
