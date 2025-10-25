import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import QClustering


class RandomGraphQClustering(QClustering):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qwedges: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * quadrangles / qwedges

    def _homogeneous_m1(self) -> Reals:
        """Compute q-clustering for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=-2)
        >>> qclust = model.nodes.qclust()
        >>> qclust.item()
        0.0924780
        >>> def compute_qclust(model):
        ...     qc = model.sample(rng=rng).struct.qclust().to_numpy()
        ...     return jnp.nanmean(jnp.asarray(qc))
        >>>
        >>> Q = jnp.array([compute_qclust(model) for _ in range(20)])
        >>> jnp.isclose(qclust, Q.mean(), rtol=1e-1).item()
        True
        >>> qc = model.nodes[:10].qclust()
        >>> qc.shape
        (10,)
        >>> jnp.all(qc == qclust).item()
        True
        """
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        """Compute q-clustering for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> mu = rng.normal(100)
        >>> model = RandomGraph(mu.size, mu=mu)
        >>> qclust = model.nodes.qclust()
        >>> qclust.shape
        (100,)
        >>> def compute_qclust(model):
        ...     qc = model.sample(rng=rng).struct.qclust().to_numpy()
        ...     return jnp.asarray(qc)
        >>>
        >>> Q = jnp.column_stack(
        ...     [compute_qclust(model) for _ in range(100)]
        ... )
        >>> Q = jnp.nanmean(Q, axis=1)
        >>> jnp.allclose(qclust, Q, rtol=1e-1).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> qc = model.nodes[vids].qclust()
        >>> qc.shape
        (4,)
        >>> jnp.allclose(qc, qclust[vids]).item()
        True
        """
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo estimate of q-clustering for undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> qc0 = model.nodes.qclust()
        >>> qc1 = model.nodes.qclust(mc=300, rng=rng)
        >>> err = jnp.linalg.norm(qc0 - qc1) / jnp.linalg.norm(qc0)
        >>> (err < 0.02).item()
        True
        >>> cor = jnp.corrcoef(qc0, qc1)[0, 1]
        >>> (cor > 0.95).item()
        True
        """
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphQClustering) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2 = stat.split_options(2, repeat=1, average=True)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qwedges = stat.nodes.motifs.qwedge(**kw2)
    return stat.m1_from_motifs(quadrangles, qwedges)
