from typing import TYPE_CHECKING, TypeVar

from grgg.models.ergm.abc import ErgmSample
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraphSampler

if TYPE_CHECKING:
    from .models import AbstractUndirectedRandomGraph
    from .views import AbstractUndirectedRandomGraphNodeView

__all__ = ("AbstractUndirectedRandomGraphSampler",)


T = TypeVar("T", bound="AbstractUndirectedRandomGraph")
V = TypeVar("V", bound="AbstractUndirectedRandomGraphNodeView")
S = TypeVar("S", bound="ErgmSample")


class AbstractUndirectedRandomGraphSampler[T, V, S](
    AbstractRandomGraphSampler[T, V, S]
):
    """Sampler for undirected random graph models."""

    nodes: V
