from typing import Any

from grgg._typing import Reals
from grgg.statistics import TClusteringStatistic


class UndirectedRandomGraphTClusteringStatistic(TClusteringStatistic):
    """Degree statistic for undirected random graphs."""

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2 = self.split_compute_kwargs(**kwargs)
        triangles = self.nodes.motifs.triangle(**kw1)
        twedges = self.nodes.motifs.twedge(**kw2)
        return 2 * triangles / twedges

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Compute t-clustering for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = UndirectedRandomGraph(100, mu=-2)
        >>> tclust = model.nodes.tclust()
        >>> tclust.item()
        0.11920292
        >>> def compute_tclust(model):
        ...     tc = model.sample(rng=rng).G.transitivity_local_undirected()
        ...     return jnp.nanmean(jnp.asarray(tc))
        >>>
        >>> T = jnp.array([compute_tclust(model) for _ in range(20)])
        >>> jnp.isclose(tclust, T.mean(), rtol=1e-1).item()
        True
        >>> tc = model.nodes[:10].tclust()
        >>> tc.shape
        (10,)
        >>> jnp.all(tc == tclust).item()
        True
        """
        return self._m1()

    def _heterogeneous_m1(
        self,
        *,
        batch_size: int | None = None,
        **kwargs: Any,  # noqa
    ) -> Reals:
        """Compute t-clustering for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = UndirectedRandomGraph(mu.size, mu=mu)
        >>> tclust = model.nodes.tclust()
        >>> tclust.shape
        (100,)
        >>> def compute_tclust(model):
        ...     tc = model.sample(rng=rng).G.transitivity_local_undirected()
        ...     return jnp.asarray(tc)
        >>>
        >>> T = jnp.column_stack(
        ...     [compute_tclust(model) for _ in range(20)]
        ... ).mean(axis=1)
        >>> jnp.allclose(tclust, T, rtol=1e-1).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> tc = model.nodes[vids].tclust()
        >>> tc.shape
        (4,)
        >>> jnp.allclose(tc, tclust[vids]).item()
        True
        """
        return self._m1(batch_size=batch_size, **kwargs)
