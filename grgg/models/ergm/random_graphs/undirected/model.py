from typing import ClassVar

import equinox as eqx

from grgg._typing import Real, RealVector
from grgg.models.abc import AbstractParameters
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraph, Mu

from .functions import RandomGraphFunctions
from .views import RandomGraphNodePairView, RandomGraphNodeView

__all__ = ("RandomGraph",)


class RandomGraph(AbstractRandomGraph):
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

    class Parameters(AbstractParameters):
        mu: Mu = eqx.field(
            default_factory=lambda: Mu(),
            converter=lambda mu: mu if isinstance(mu, Mu) else Mu(mu),
        )

    n_nodes: int = eqx.field(static=True)
    parameters: Parameters

    is_directed: ClassVar[bool] = False
    functions: ClassVar[type[RandomGraphFunctions]] = RandomGraphFunctions

    nodes_cls: ClassVar[type[RandomGraphNodeView]] = RandomGraphNodeView
    pairs_cls: ClassVar[type[RandomGraphNodePairView]] = RandomGraphNodePairView

    def __init__(
        self,
        n_nodes: int,
        *,
        parameters: Parameters | None = None,
        **kwargs: Real | RealVector,
    ) -> None:
        self.n_nodes = n_nodes
        self.parameters = self._make_parameters(parameters, **kwargs)

    def _equals(self, other: object) -> bool:
        return super()._equals(other)
