from typing import ClassVar, TypeVar

import equinox as eqx

from grgg._typing import Real, RealVector
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraph, Mu

from .functions import RandomGraphCoupling
from .views import RandomGraphNodePairView, RandomGraphNodeView

__all__ = ("RandomGraph",)


V = TypeVar("V", bound=RandomGraphNodeView)
E = TypeVar("E", bound=RandomGraphNodePairView)


class RandomGraph[V, E](AbstractRandomGraph[V, E]):
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
    coupling: RandomGraphCoupling = eqx.field(init=False)

    is_directed: ClassVar[bool] = False
    nodes_cls: ClassVar[type[RandomGraphNodeView]] = RandomGraphNodeView
    pairs_cls: ClassVar[type[RandomGraphNodePairView]] = RandomGraphNodePairView

    def __init__(
        self,
        n_nodes: int,
        mu: Real | RealVector | Mu | None = None,
    ) -> None:
        self.n_nodes = n_nodes
        self.mu = mu if isinstance(mu, Mu) else Mu(mu)
        self.coupling = self._init_coupling()

    def _repr_inner(self) -> str:
        return f"{self.n_nodes}, {self.mu}"

    @property
    def parameters(self) -> dict[str, Mu]:
        return {"mu": self.mu}

    def _init_coupling(self) -> RandomGraphCoupling:
        return RandomGraphCoupling()

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.n_nodes == other.n_nodes
            and self.mu.equals(other.mu)
            and self.coupling.equals(other.coupling)
        )
