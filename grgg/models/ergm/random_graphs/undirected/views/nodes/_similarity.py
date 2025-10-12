from typing import Any

from grgg._typing import Reals
from grgg.statistics import StructuralSimilarity


class UndirectedRandomGraphStructuralSimilarity(StructuralSimilarity):
    """Structural similarity statistic for undirected random graphs."""

    @staticmethod
    def m1_from_motifs(triangles: Reals, twedges: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 4 * triangles / (twedges + theads)

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2, kw3 = self.split_compute_kwargs(3, same_seed=True, **kwargs)
        triangles = self.nodes.motifs.triangle(**kw1)
        twedges = self.nodes.motifs.twedge(**kw2)
        theads = self.nodes.motifs.thead(**kw3)
        return self.m1_from_motifs(triangles, twedges, theads)

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
        return self._m1(**kwargs)

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
        return self._m1(**kwargs)
