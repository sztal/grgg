from typing import Any

import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import QClosure


class UndirectedRandomGraphQClosure(QClosure):
    """Quadrangle closure statistic for undirected random graphs."""

    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * quadrangles / qheads

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Compute q-closure for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = UndirectedRandomGraph(100, mu=-2)
        >>> qclosure = model.nodes.qclosure()
        >>> qclosure.item()
        0.0924780
        >>> def compute_qclosure(model):
        ...     qc = model.sample(rng=rng).struct.qclosure().to_numpy()
        ...     return jnp.nanmean(jnp.asarray(qc))
        >>>
        >>> Q = jnp.array([compute_qclosure(model) for _ in range(20)])
        >>> jnp.isclose(qclosure, Q.mean(), rtol=1e-1).item()
        True
        >>> qc = model.nodes[:10].qclosure()
        >>> qc.shape
        (10,)
        >>> jnp.all(qc == qclosure).item()
        True
        """
        return _m1(self, **kwargs)

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Compute q-closure for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import UndirectedRandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> mu = rng.normal(100) - 1
        >>> model = UndirectedRandomGraph(mu.size, mu=mu)
        >>> qclosure = model.nodes.qclosure()
        >>> qclosure.shape
        (100,)
        >>> def compute_qclosure(model):
        ...     qc = model.sample(rng=rng).struct.qclosure().to_numpy()
        ...     return jnp.asarray(qc)
        >>>
        >>> Q = jnp.column_stack(
        ...     [compute_qclosure(model) for _ in range(100)]
        ... )
        >>> Q = jnp.nanmean(Q, axis=1)
        >>> jnp.allclose(qclosure, Q, rtol=1e-1, atol=1e-2).item()
        True
        >>> vids = jnp.array([0, 11, 27, 89])
        >>> qc = model.nodes[vids].qclosure()
        >>> qc.shape
        (4,)
        >>> jnp.allclose(qc, qclosure[vids]).item()
        True
        """
        return _m1(self, **kwargs)


@eqx.filter_jit
def _m1(stat: UndirectedRandomGraphQClosure, **kwargs: Any) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2 = stat.split_compute_kwargs(same_seed=True, **kwargs)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qheads = stat.nodes.motifs.qhead(**kw2)
    return stat.m1_from_motifs(quadrangles, qheads)
