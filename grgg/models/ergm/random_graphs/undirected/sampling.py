from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc import ErgmSample
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraphSampler

if TYPE_CHECKING:
    from .model import RandomGraph
    from .views import RandomGraphNodeView

__all__ = ("RandomGraphSampler",)


T = TypeVar("T", bound="RandomGraph")
V = TypeVar("V", bound="RandomGraphNodeView")
S = TypeVar("S", bound="ErgmSample")


class RandomGraphSampler[T, V, S](AbstractRandomGraphSampler[T, V, S]):
    """Sampler for undirected random graph models."""

    nodes: V
