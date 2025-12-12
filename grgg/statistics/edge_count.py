from typing import Any, ClassVar

import jax.numpy as jnp
from scipy.sparse import sparray, spmatrix

from grgg._typing import Reals
from grgg.statistics.abc import AbstractErgmStatistic, T
from grgg.utils.dispatch import dispatch

__all__ = ("EdgeCount",)


class EdgeCount(AbstractErgmStatistic):
    """Edge count statistic for ERGMs."""

    module: T

    supported_moments: ClassVar[tuple[int, ...]] = (1,)
    supports_monte_carlo: ClassVar[bool] = False

    def _homogeneous_m1(self, **kwargs) -> Reals:  # noqa
        """Expected edge count for homogeneous random graph models."""
        ecount = self.model.nodes.degree(**kwargs) * self.model.n_nodes
        if self.model.is_undirected:
            ecount = ecount / 2
        return ecount

    def _heterogeneous_m1_exact(self, **kwargs) -> Reals:  # noqa
        """Expected edge count for heterogeneous random graph models."""
        ecount = self.model.nodes.degree(**kwargs).sum()
        if self.model.is_undirected:
            ecount = ecount / 2
        return ecount

    @dispatch
    def _observed(self, _, obj: Any) -> Reals:
        obj = self.validate_object(obj)
        ecount = jnp.sum(obj)
        if self.model.is_undirected:
            ecount = ecount / 2
        return ecount

    @_observed.dispatch
    def _(self, _, obj: spmatrix | sparray) -> Reals:
        obj = self.validate_object(obj)
        ecount = jnp.asarray(obj.data.sum())
        if self.model.is_undirected:
            ecount = ecount / 2
        return ecount

    try:
        import igraph as ig

        @_observed.dispatch
        def _(self, _, obj: ig.Graph) -> Reals:
            obj = self.validate_object(obj)
            return jnp.asarray(obj.ecount())

    except ImportError:
        pass

    try:
        import networkx as nx

        @_observed.dispatch
        def _(self, _, obj: nx.Graph | nx.DiGraph) -> Reals:
            obj = self.validate_object(obj)
            return jnp.asarray(obj.number_of_edges())

    except ImportError:
        pass
