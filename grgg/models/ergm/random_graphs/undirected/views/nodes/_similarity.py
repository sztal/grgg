from typing import Any

import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import StructuralSimilarity


class UndirectedRandomGraphStructuralSimilarity(StructuralSimilarity):
    """Structural similarity statistic for undirected random graphs."""

    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(triangles: Reals, twedges: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 4 * triangles / (twedges + theads)

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Compute similarity for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = UndirectedRandomGraph(100, mu=-2)
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
        return _m1(self, **kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Compute similarity for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = UndirectedRandomGraph(mu.size, mu=mu)
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
        return _m1(self, **kwargs)


@eqx.filter_jit
def _m1(stat: UndirectedRandomGraphStructuralSimilarity, **kwargs: Any) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_compute_kwargs(3, same_seed=True, **kwargs)
    triangles = stat.nodes.motifs.triangle(**kw1)
    twedges = stat.nodes.motifs.twedge(**kw2)
    theads = stat.nodes.motifs.thead(**kw3)
    return stat.m1_from_motifs(triangles, twedges, theads)
