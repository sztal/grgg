from functools import singledispatchmethod
from typing import Any, ClassVar

import jax.numpy as jnp
from scipy.sparse import sparray, spmatrix

from grgg._typing import Reals
from grgg.statistics.abc import VT, AbstractErgmNodeStatistic


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

    label: ClassVar[str] = "degree"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)
    supports_monte_carlo: ClassVar[bool] = False

    @singledispatchmethod
    def observed(self, obj: Any, *args: Any, **kwargs: Any) -> Reals:
        """Compute the observed value of the statistic for a given object."""
        obj = jnp.asarray(self.validate_object(obj))
        return self.observed(obj, *args, **kwargs)

    @observed.register
    def _(self, obj: jnp.ndarray, *args: Any, **kwargs: Any) -> Reals:  # noqa
        obj = self.validate_object(obj)
        return jnp.sum(obj, axis=-1)

    @observed.register
    def _(self, obj: sparray, *args: Any, **kwargs: Any) -> Reals:  # noqa
        obj = self.validate_object(obj)
        return jnp.asarray(obj.sum(axis=-1))

    @observed.register
    def _(self, obj: spmatrix, *args: Any, **kwargs: Any) -> Reals:  # noqa
        obj = self.validate_object(obj)
        return jnp.asarray(obj.sum(axis=-1)).ravel()

    try:
        import igraph as ig

        @observed.register
        def _(self, obj: ig.Graph, *args: Any, **kwargs: Any) -> Reals:  # noqa
            obj = self.validate_object(obj)
            return jnp.asarray(obj.degree(*args, **kwargs))

    except ImportError:
        pass

    try:
        import networkx as nx

        @observed.register
        def _(self, obj: nx.Graph, *args: Any, **kwargs: Any) -> Reals:  # noqa
            obj = self.validate_object(obj)
            return jnp.asarray([d for _, d in obj.degree(*args, **kwargs)])

        @observed.register
        def _(self, obj: nx.DiGraph, *args: Any, **kwargs: Any) -> Reals:  # noqa
            obj = self.validate_object(obj)
            return jnp.asarray([d for _, d in obj.degree(*args, **kwargs)])

    except ImportError:
        pass
