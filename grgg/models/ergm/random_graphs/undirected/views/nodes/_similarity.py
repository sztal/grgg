import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import StructuralSimilarity


class RandomGraphStructuralSimilarity(StructuralSimilarity):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(triangles: Reals, twedges: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 4 * triangles / (twedges + theads)

    def _homogeneous_m1(self) -> Reals:
        """Compute similarity for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=-2)
        >>> sim = model.nodes.similarity()
        >>> sim.item()
        0.1192029
        >>> def compute_sim(model):
        ...     s = model.sample(rng=rng).struct.similarity().to_numpy()
        ...     return jnp.nanmean(jnp.asarray(s))
        >>>
        >>> S = jnp.array([compute_sim(model) for _ in range(20)])
        >>> jnp.isclose(sim, S.mean(), rtol=1e-1).item()
        True
        >>> s = model.nodes[:10].similarity()
        >>> s.shape
        (10,)
        >>> jnp.all(s == sim).item()
        True
        """
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        """Compute similarity for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = RandomGraph(mu.size, mu=mu)
        >>> sim = model.nodes.similarity()
        >>> sim.shape
        (100,)
        >>> def compute_sim(model):
        ...     s = model.sample(rng=rng).struct.similarity().to_numpy()
        ...     return jnp.nanmean(jnp.asarray(s))
        >>>
        >>> S = jnp.array([compute_sim(model) for _ in range(20)])
        >>> jnp.isclose(sim.mean(), S.mean(), rtol=1e-1).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> s = model.nodes[vids].similarity()
        >>> s.shape
        (4,)
        >>> jnp.allclose(s, sim[vids], rtol=1e-1).item()
        True
        """
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo estimate of similarity for undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> s0 = model.nodes.similarity()
        >>> s1 = model.nodes.similarity(mc=300, rng=rng)
        >>> err = jnp.linalg.norm(s0 - s1) / jnp.linalg.norm(s0)
        >>> (err < 0.02).item()
        True
        >>> cor = jnp.corrcoef(s0, s1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphStructuralSimilarity) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_options(3, repeat=1, average=True)
    triangles = stat.nodes.motifs.triangle(**kw1)
    twedges = stat.nodes.motifs.twedge(**kw2)
    theads = stat.nodes.motifs.thead(**kw3)
    return stat.m1_from_motifs(triangles, twedges, theads)
