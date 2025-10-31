from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc.motifs import (
    AbstractErgmNodeMotifs,
    AbstractErgmNodePairMotifs,
)

if TYPE_CHECKING:
    from .model import AbstractRandomGraph
    from .views import AbstractRandomGraphNodePairView, AbstractRandomGraphNodeView

    T = TypeVar("T", bound="AbstractRandomGraph")
    E = TypeVar("E", bound=AbstractRandomGraphNodePairView[T])
    V = TypeVar("V", bound=AbstractRandomGraphNodeView[T])

__all__ = (
    "AbstractRandomGraphNodeMotifs",
    "AbstractRandomGraphNodePairMotifs",
)


class AbstractRandomGraphNodeMotifs[T](AbstractErgmNodeMotifs[T]):
    """Abstract base class for random graph node motif statistics."""


class AbstractRandomGraphNodePairMotifs[T](AbstractErgmNodePairMotifs[T]):
    """Abstract base class for random graph node pair motif statistics."""
