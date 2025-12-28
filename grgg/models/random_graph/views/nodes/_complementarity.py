import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import StructuralComplementarity


class RandomGraphStructuralComplementarity(StructuralComplementarity):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qwedges: Reals, qheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 4 * quadrangles / (qwedges + qheads)

    def _homogeneous_m1(self) -> Reals:
        """Compute complementarity for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=-2)
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
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        """Compute complementarity for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> mu = rng.normal(100)
        >>> model = RandomGraph(mu.size, mu=mu)
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
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo estimate of complementarity for undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> n = 500
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> c0 = model.nodes.complementarity()
        >>> c1 = model.nodes.complementarity(mc=100, repeat=5, rng=rng)
        >>> err = jnp.linalg.norm(c0 - c1) / jnp.linalg.norm(c0)
        >>> (err < 0.05).item()
        True
        >>> cor = jnp.corrcoef(c0, c1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphStructuralComplementarity) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2, kw3 = stat.split_options(3, repeat=1, average=True)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qwedges = stat.nodes.motifs.qwedge(**kw2)
    qheads = stat.nodes.motifs.qhead(**kw3)
    return stat.m1_from_motifs(quadrangles, qwedges, qheads)
