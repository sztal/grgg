from functools import singledispatchmethod
from typing import Any, ClassVar

import jax.numpy as jnp
from scipy.sparse import sparray

from grgg._typing import Reals
from grgg.statistics.abc import VT, AbstractErgmNodeStatistic


class Degree(AbstractErgmNodeStatistic):
    module: VT

    label: ClassVar[str] = "degree"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)
    supports_monte_carlo: ClassVar[bool] = False

    @singledispatchmethod
    def observed(self, obj: Any, *args: Any, **kwargs: Any) -> Reals:
        """Compute the observed value of the statistic for a given object."""
        self.check_observed(obj)
        return self.observed(jnp.asarray(obj), *args, **kwargs)

    @observed.register
    def _(self, obj: jnp.ndarray, *args: Any, **kwargs: Any) -> Reals:  # noqa
        self.check_observed(obj)
        return jnp.sum(obj, axis=-1)

    @observed.register
    def _(self, obj: sparray, *args: Any, **kwargs: Any) -> Reals:  # noqa
        self.check_observed(obj)
        return jnp.asarray(obj.sum(axis=-1))

    @observed.register
    def _(self, obj: jnp.ndarray, *args: Any, **kwargs: Any) -> Reals:  # noqa
        self.check_observed(obj)
        return jnp.sum(obj, axis=-1)

    try:
        import igraph as ig

        @observed.register
        def _(self, obj: ig.Graph, *args: Any, **kwargs: Any) -> Reals:  # noqa
            self.check_observed(obj)
            return jnp.asarray(obj.degree(*args, **kwargs))

    except ImportError:
        pass

    try:
        import networkx as nx

        @observed.register
        def _(self, obj: nx.Graph, *args: Any, **kwargs: Any) -> Reals:  # noqa
            self.check_observed(obj)
            return jnp.asarray([d for _, d in obj.degree(*args, **kwargs)])

        @observed.register
        def _(self, obj: nx.DiGraph, *args: Any, **kwargs: Any) -> Reals:  # noqa
            self.check_observed(obj)
            return jnp.asarray([d for _, d in obj.degree(*args, **kwargs)])

    except ImportError:
        pass
