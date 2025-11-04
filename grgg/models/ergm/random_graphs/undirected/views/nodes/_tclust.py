import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import TClustering


class RandomGraphTClustering(TClustering):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(triangles: Reals, twedges: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * triangles / twedges

    def _homogeneous_m1(self) -> Reals:  # noqa
        """Compute t-clustering for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=-2)
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
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        """Compute t-clustering for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = RandomGraph(mu.size, mu=mu)
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
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo estimate of t-clustering for undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> tc0 = model.nodes.tclust()
        >>> tc1 = model.nodes.tclust(mc=300, repeat=5, rng=rng)
        >>> err = jnp.linalg.norm(tc0 - tc1) / jnp.linalg.norm(tc0)
        >>> (err < 0.03).item()
        True
        >>> cor = jnp.corrcoef(tc0, tc1)[0, 1]
        >>> (cor > 0.90).item()
        True
        """
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphTClustering) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2 = stat.split_options(2, repeat=1, average=True)
    triangles = stat.nodes.motifs.triangle(**kw1)
    twedges = stat.nodes.motifs.twedge(**kw2)
    return stat.m1_from_motifs(triangles, twedges)
