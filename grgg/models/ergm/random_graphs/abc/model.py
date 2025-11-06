from typing import Any

import equinox as eqx

from grgg.models.ergm.abc import AbstractErgm

from .functions import AbstractRandomGraphFunctions

__all__ = ("AbstractRandomGraph",)


class AbstractRandomGraph(AbstractErgm):
    """Abstract base class for random graph models."""

    n_nodes: eqx.AbstractVar[int]
    functions: eqx.AbstractClassVar[type[AbstractRandomGraphFunctions]]

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_nodes

    # Model functions ----------------------------------------------------------------

    def edge_density(self, *args: Any, **kwargs: Any) -> float:
        """Expected edge density of the model."""
        return self.nodes.edge_density(*args, **kwargs)
