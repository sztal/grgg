from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.random_graphs.abc import (
    AbstractRandomGraphNodePairView,
    AbstractRandomGraphNodeView,
)

if TYPE_CHECKING:
    from .models import AbstractUndirectedRandomGraph

__all__ = (
    "AbstractUndirectedRandomGraphNodeView",
    "AbstractUndirectedRandomGraphNodePairView",
)


T = TypeVar("T", bound="AbstractUndirectedRandomGraph")


class AbstractUndirectedRandomGraphNodeView[T, V](AbstractRandomGraphNodeView[T, V]):
    """Node view for undirected random graph models."""

    model: T


class AbstractUndirectedRandomGraphNodePairView[T, E](
    AbstractRandomGraphNodePairView[T, E]
):
    """Node pair view for undirected random graph models."""

    model: T
