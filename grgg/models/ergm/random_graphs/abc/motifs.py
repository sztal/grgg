from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc.motifs import (
    AbstractErgmNodeMotifs,
    AbstractErgmNodePairMotifs,
)

if TYPE_CHECKING:
    from .models import AbstractRandomGraph
    from .views import AbstractRandomGraphNodePairView, AbstractRandomGraphNodeView

    T = TypeVar("T", bound="AbstractRandomGraph")
    E = TypeVar("E", bound=AbstractRandomGraphNodePairView[T])
    V = TypeVar("V", bound=AbstractRandomGraphNodeView[T])

__all__ = (
    "AbstractRandomGraphNodeMotifs",
    "AbstractRandomGraphNodePairMotifs",
)


class AbstractRandomGraphNodeMotifs[V](AbstractErgmNodeMotifs[V]):
    """Abstract base class for random graph node motif statistics."""


class AbstractRandomGraphNodePairMotifs[E](AbstractErgmNodePairMotifs[E]):
    """Abstract base class for random graph node pair motif statistics."""
