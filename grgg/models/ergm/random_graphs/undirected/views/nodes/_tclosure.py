import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import TClosure


class RandomGraphTClosure(TClosure):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(triangles: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * triangles / theads

    def _homogeneous_m1(self) -> Reals:
        """Compute t-closure for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
        >>> tclosure = model.nodes.tclosure()
        >>> tclosure.item()
        0.11920292
        >>> def compute_tclosure(model):
        ...     tc = model.sample(rng=rng).struct.tclosure().to_numpy()
        ...     return jnp.nanmean(jnp.asarray(tc))
        >>>
        >>> T = jnp.array([compute_tclosure(model) for _ in range(20)])
        >>> jnp.isclose(tclosure, T.mean(), rtol=1e-1).item()
        True
        >>> tc = model.nodes[:10].tclosure()
        >>> tc.shape
        (10,)
        >>> jnp.all(tc == tclosure).item()
        True
        """
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        """Compute t-closure for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> mu = rng.normal(100) - 1
        >>> model = RandomGraph(mu.size, mu=mu)
        >>> tclosure = model.nodes.tclosure()
        >>> tclosure.shape
        (100,)
        >>> def compute_tclosure(model):
        ...     tc = model.sample(rng=rng).struct.tclosure().to_numpy()
        ...     return jnp.asarray(tc)
        >>>
        >>> T = jnp.column_stack(
        ...     [compute_tclosure(model) for _ in range(200)]
        ... )
        >>> T = jnp.nanmean(T, axis=1)
        >>> jnp.allclose(tclosure, T, rtol=2e-1, atol=5e-2).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> tc = model.nodes[vids].tclosure()
        >>> tc.shape
        (4,)
        >>> jnp.allclose(tc, tclosure[vids]).item()
        True
        """
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo estimate of t-closure for undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 500
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> tc0 = model.nodes.tclosure()
        >>> tc1 = model.nodes.tclosure(mc=50, repeat=5, rng=rng)
        >>> err = jnp.linalg.norm(tc0 - tc1) / jnp.linalg.norm(tc0)
        >>> (err < 0.02).item()
        True
        >>> cor = jnp.corrcoef(tc0, tc1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphTClosure) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2 = stat.split_options(2, repeat=1, average=True)
    triangles = stat.nodes.motifs.triangle(**kw1)
    theads = stat.nodes.motifs.thead(**kw2)
    return stat.m1_from_motifs(triangles, theads)
