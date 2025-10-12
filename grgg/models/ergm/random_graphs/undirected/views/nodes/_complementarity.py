from typing import Any

from grgg._typing import Reals
from grgg.statistics import StructuralComplementarity


class UndirectedRandomGraphStructuralComplementarity(StructuralComplementarity):
    """Structural complementarity statistic for undirected random graphs."""

    @staticmethod
    def m1_from_motifs(quadrangles: Reals, qwedges: Reals, qheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 4 * quadrangles / (qwedges + qheads)

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2, kw3 = self.split_compute_kwargs(3, same_seed=True, **kwargs)
        quadrangles = self.nodes.motifs.quadrangle(**kw1)
        qwedges = self.nodes.motifs.qwedge(**kw2)
        qheads = self.nodes.motifs.qhead(**kw3)
        return self.m1_from_motifs(quadrangles, qwedges, qheads)

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Compute complementarity for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = UndirectedRandomGraph(100, mu=-2)
        >>> comp = model.nodes.complementarity()
        >>> comp.item()
        0.0924780
        >>> def compute_comp(model):
        ...     s = model.sample(rng=rng).struct.complementarity().to_numpy()
        ...     return jnp.nanmean(jnp.asarray(s))
        >>>
        >>> C = jnp.array([compute_comp(model) for _ in range(20)])
        >>> jnp.isclose(comp, C.mean(), rtol=1e-1).item()
        True
        >>> c = model.nodes[:10].complementarity()
        >>> c.shape
        (10,)
        >>> jnp.all(c == comp).item()
        True
        """
        return self._m1(**kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Compute complementarity for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = UndirectedRandomGraph(mu.size, mu=mu)
        >>> comp = model.nodes.complementarity()
        >>> comp.shape
        (100,)
        >>> def compute_comp(model):
        ...     s = model.sample(rng=rng).struct.complementarity().to_numpy()
        ...     return jnp.asarray(s)
        >>>
        >>> C = jnp.array([compute_comp(model) for _ in range(20)])
        >>> jnp.isclose(comp.mean(), C.mean(), rtol=1e-1).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> s = model.nodes[vids].complementarity()
        >>> s.shape
        (4,)
        >>> jnp.allclose(s, comp[vids], rtol=1e-1).item()
        True
        """
        return self._m1(**kwargs)
