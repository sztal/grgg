from abc import abstractmethod
from typing import ClassVar, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Real, RealVector
from grgg.models.ergm.random_graphs.abc import AbstractRandomGraph

from .functions import UndirectedRandomGraphCoupling
from .parameters import Mu, UndirectedRandomGraphParameters
from .sampling import AbstractUndirectedRandomGraphSampler
from .views import AbstractRandomGraphNodePairView, AbstractRandomGraphNodeView

__all__ = ("AbstractUndirectedRandomGraph",)


T = TypeVar("T", bound="AbstractUndirectedRandomGraph")
P = TypeVar("P", bound=UndirectedRandomGraphParameters)
V = TypeVar("V", bound=AbstractRandomGraphNodeView)
E = TypeVar("E", bound=AbstractRandomGraphNodePairView)
S = TypeVar("S", bound=AbstractUndirectedRandomGraphSampler)


class AbstractUndirectedRandomGraph[T, P, V, E, S](AbstractRandomGraph[T, P, V, E, S]):
    """Abstract base class for undirected random graph models."""

    is_directed: ClassVar[bool] = False
    coupling: eqx.AbstractVar[UndirectedRandomGraphCoupling]

    parameters_cls: eqx.AbstractClassVar[type[P]]

    @abstractmethod
    def __init__(
        self,
        n_nodes: int,
        parameters: P | jnp.ndarray | Mu | None = None,
        *,
        mu: Real | RealVector | None = None,
    ) -> None:
        if mu is not None and parameters is not None:
            errmsg = "Cannot pass both 'mu' and 'parameters'."
            raise ValueError(errmsg)
        if mu is not None:
            parameters = mu
        self.n_nodes = n_nodes
        self.parameters = UndirectedRandomGraphParameters.from_arrays(parameters)
        self.coupling = self._init_coupling()

    def _repr_inner(self) -> str:
        return f"{self.n_nodes}, {self.mu}"

    @property
    def mu(self) -> Mu:
        """Node parameter."""
        return self.parameters.mu

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.n_nodes == other.n_nodes
            and self.mu.equals(other.mu)
            and self.coupling.equals(other.coupling)
        )
