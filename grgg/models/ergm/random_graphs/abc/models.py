from abc import abstractmethod
from typing import Any, TypeVar

from grgg._typing import Reals
from grgg.models.ergm.abc import AbstractErgm
from grgg.utils.misc import sigmoid

from .functions import AbstractCoupling
from .sampling import AbstractRandomGraphSampler
from .views import AbstractRandomGraphNodePairView, AbstractRandomGraphNodeView

__all__ = ("AbstractRandomGraph",)


T = TypeVar("T", bound="AbstractRandomGraph")
V = TypeVar("V", bound=AbstractRandomGraphNodeView)
E = TypeVar("E", bound=AbstractRandomGraphNodePairView)
S = TypeVar("S", bound=AbstractRandomGraphSampler)


class AbstractRandomGraph[T, V, E, S](AbstractErgm[T, V, E, S]):
    """Abstract base class for random graph models."""

    coupling: AbstractCoupling

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compute connection probabilities from model parameters."""
        couplings = self.coupling(*args, **kwargs)
        return self.probability(couplings)

    def probability(self, coupling: Reals) -> Reals:
        """Connection probability function.

        Parameters
        ----------
        coupling
            Node pair couplings.
        """
        return sigmoid(coupling)

    def edge_density(self, *args: Any, **kwargs: Any) -> float:
        """Expected edge density of the model."""
        return self.nodes.degree(*args, **kwargs).mean() / (self.n_nodes - 1)

    @abstractmethod
    def _init_coupling(self) -> AbstractCoupling:
        """Initialize the coupling function."""
