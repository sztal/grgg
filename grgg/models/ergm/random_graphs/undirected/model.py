from typing import ClassVar, TypeVar

import equinox as eqx

from .abc import AbstractUndirectedRandomGraph, Mu
from .abc.functions import UndirectedRandomGraphCoupling
from .sampling import UndirectedRandomGraphSampler
from .views import UndirectedRandomGraphNodePairView, UndirectedRandomGraphNodeView

__all__ = ("UndirectedRandomGraph",)


T = TypeVar("T", bound="UndirectedRandomGraph")
V = TypeVar("V", bound=UndirectedRandomGraphNodeView)
E = TypeVar("E", bound=UndirectedRandomGraphNodePairView)
S = TypeVar("S", bound=UndirectedRandomGraphSampler)


class UndirectedRandomGraph[T, V, E, S](AbstractUndirectedRandomGraph[T, V, E, S]):
    """Undirected random graph model.

    It is equivalent to the `(n, p)`-Erdős–Rényi model when `mu` is homogeneous,
    or to the soft configuration model when `mu` is heterogeneous.

    Attributes
    ----------
    n_nodes
        Number of nodes.
    mu
        Parameter controlling the expected degree of nodes.
    """

    n_nodes: int = eqx.field(static=True)
    mu: Mu
    coupling: UndirectedRandomGraphCoupling = eqx.field(init=False)

    node_view_cls: ClassVar[type[V]] = UndirectedRandomGraphNodeView  # type: ignore
    pair_view_cls: ClassVar[type[E]] = UndirectedRandomGraphNodePairView  # type: ignore
    sampler_cls: ClassVar[type[S]] = UndirectedRandomGraphSampler  # type: ignore

    _node_view_type: ClassVar[
        type[UndirectedRandomGraphNodeView]
    ] = UndirectedRandomGraphNodeView

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the model has heterogeneous parameters."""
        return self.mu.is_heterogeneous

    def _init_coupling(self) -> UndirectedRandomGraphCoupling:
        return UndirectedRandomGraphCoupling()
