from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.random_graphs.abc import AbstractRandomGraphSampler

if TYPE_CHECKING:
    from .model import RandomGraph
    from .views import RandomGraphNodeView

__all__ = ("RandomGraphSampler",)


T = TypeVar("T", bound="RandomGraph")


class RandomGraphSampler[T](AbstractRandomGraphSampler[T]):
    """Sampler for undirected random graph models."""

    nodes: "RandomGraphNodeView"
