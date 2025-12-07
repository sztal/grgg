from typing import Any, ClassVar

import jax.numpy as jnp
from scipy.sparse import sparray, spmatrix

from grgg._typing import Reals
from grgg.statistics.abc import VT, AbstractErgmNodeStatistic
from grgg.utils.dispatch import dispatch

__all__ = ("Degree",)


class Degree(AbstractErgmNodeStatistic):
    """Node degree statistic for ERGMs.

    Examples
    --------
    Statistics can be computed not only as model expectatiosns but also
    as observed values on given objects.
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix, csr_array
    >>> from grgg import RandomGraph
    >>> n = 10
    >>> model = RandomGraph(n)
    >>> A = np.ones((n, n)) - np.eye(n)
    >>> model.nodes.degree.observed(A)
    Array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])
    >>> model.nodes.degree.observed(A.tolist())
    Array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])
    >>> A_sparse = csr_matrix(A)
    >>> model.nodes.degree.observed(A_sparse)
    Array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])
    >>> A_sparse = csr_array(A)
    >>> model.nodes.degree.observed(A_sparse)
    Array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])
    >>> import networkx as nx
    >>> G = nx.complete_graph(n)
    >>> model.nodes.degree.observed(G)
    Array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])
    >>> import igraph as ig
    >>> G_ig = ig.Graph.Full(n)
    >>> model.nodes.degree.observed(G_ig)
    Array([9., 9., 9., 9., 9., 9., 9., 9., 9., 9.])
    """

    module: VT

    supported_moments: ClassVar[tuple[int, ...]] = (1,)
    supports_monte_carlo: ClassVar[bool] = False

    @dispatch
    def _observed(self, model: Any, obj: Any, *args: Any, **kwargs: Any) -> Reals:
        return super()._observed(model, obj, *args, **kwargs)

    @_observed.dispatch
    def _(self, _, obj: jnp.ndarray) -> Reals:
        obj = self.validate_object(obj)
        return jnp.sum(obj, axis=-1)

    @_observed.dispatch
    def _(self, _, obj: spmatrix | sparray) -> Reals:
        obj = self.validate_object(obj)
        return jnp.asarray(obj.sum(axis=-1)).ravel()

    try:
        import igraph as ig

        @_observed.dispatch
        def _(self, _, obj: ig.Graph, *args: Any, **kwargs: Any) -> Reals:
            obj = self.validate_object(obj)
            return jnp.asarray(obj.degree(*args, **kwargs))

    except ImportError:
        pass

    try:
        import networkx as nx

        @_observed.dispatch
        def _(self, _, obj: nx.Graph | nx.DiGraph, *args: Any, **kwargs: Any) -> Reals:
            obj = self.validate_object(obj)
            return jnp.asarray([d for _, d in obj.degree(*args, **kwargs)])

    except ImportError:
        pass
