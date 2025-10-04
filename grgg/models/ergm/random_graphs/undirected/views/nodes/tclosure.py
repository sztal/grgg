from typing import Any

from grgg._typing import Reals
from grgg.statistics import TClosure


class UndirectedRandomGraphTClosure(TClosure):
    """Triangle closure statistic for undirected random graphs."""

    @staticmethod
    def m1_from_motifs(triangles: Reals, theads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * triangles / theads

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2 = self.split_compute_kwargs(same_seed=True, **kwargs)
        triangles = self.nodes.motifs.triangle(**kw1)
        theads = self.nodes.motifs.thead(**kw2)
        return self.m1_from_motifs(triangles, theads)

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:
        """Compute t-closure for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = UndirectedRandomGraph(100, mu=-2)
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
        return self._m1(**kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Compute t-closure for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> mu = rng.normal(100) - 1
        >>> model = UndirectedRandomGraph(mu.size, mu=mu)
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
        return self._m1(**kwargs)
