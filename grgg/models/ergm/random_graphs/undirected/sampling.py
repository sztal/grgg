from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc import ErgmSample

from .abc import AbstractUndirectedRandomGraphSampler

if TYPE_CHECKING:
    from .model import UndirectedRandomGraph
    from .views import UndirectedRandomGraphNodeView

__all__ = ("UndirectedRandomGraphSampler",)


T = TypeVar("T", bound="UndirectedRandomGraph")
V = TypeVar("V", bound="UndirectedRandomGraphNodeView")
S = TypeVar("S", bound="ErgmSample")


class UndirectedRandomGraphSampler[T, V, S](
    AbstractUndirectedRandomGraphSampler[T, V, S]
):
    """Sampler for undirected random graph models."""

    nodes: V
