from typing import Any

import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import QClustering


class UndirectedRandomGraphQClustering(QClustering):
    """Quadrangle clustering statistic for undirected random graphs."""

    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qwedges: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * quadrangles / qwedges

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Compute q-clustering for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = UndirectedRandomGraph(100, mu=-2)
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
        return _m1(self, **kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Compute q-clustering for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> mu = rng.normal(100)
        >>> model = UndirectedRandomGraph(mu.size, mu=mu)
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
        return _m1(self, **kwargs)


@eqx.filter_jit
def _m1(stat: UndirectedRandomGraphQClustering, **kwargs: Any) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2 = stat.split_compute_kwargs(same_seed=True, **kwargs)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qwedges = stat.nodes.motifs.qwedge(**kw2)
    return stat.m1_from_motifs(quadrangles, qwedges)
