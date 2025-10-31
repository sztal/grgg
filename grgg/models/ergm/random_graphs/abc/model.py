from abc import abstractmethod
from typing import Any

import equinox as eqx

from grgg._typing import Reals
from grgg.models.ergm.abc import AbstractErgm

from .functions import logprobs, probs

__all__ = ("AbstractRandomGraph",)


class AbstractRandomGraph(AbstractErgm):
    """Abstract base class for random graph models."""

    n_nodes: eqx.AbstractVar[int]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compute connection probabilities from model parameters."""
        couplings = self.coupling(*args, **kwargs)
        return self.probability(couplings)

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_nodes

    # Model functions ----------------------------------------------------------------

    def edge_density(self, *args: Any, **kwargs: Any) -> float:
        """Expected edge density of the model."""
        return self.nodes.edge_density(*args, **kwargs)

    @abstractmethod
    def couplings(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute edge couplings."""

    def logprobs(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute edge log-probabilities."""
        return logprobs(self, *args, **kwargs)

    def probs(self, *args: Any, **kwargs: Any) -> Reals:
        """Compute edge probabilities."""
        return probs(self, *args, **kwargs)
