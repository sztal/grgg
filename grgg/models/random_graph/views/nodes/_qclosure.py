import equinox as eqx

from grgg._typing import Reals
from grgg.statistics import QClosure


class RandomGraphQClosure(QClosure):
    @staticmethod
    @eqx.filter_jit
    def m1_from_motifs(quadrangles: Reals, qheads: Reals) -> Reals:
        """Compute the first moment of the statistic from motifs counts."""
        return 2 * quadrangles / qheads

    def _homogeneous_m1(self) -> Reals:
        """Compute q-closure for a homogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
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
        return _m1(self)

    def _heterogeneous_m1_exact(self) -> Reals:
        """Compute q-closure for a heterogeneous undirected random graph.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> mu = rng.normal(100) - 1
        >>> model = RandomGraph(mu.size, mu=mu)
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
        return _m1(self)

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo estimate for q-closure for undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(303)
        >>> n = 500
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> qc0 = model.nodes.qclosure()
        >>> qc1 = model.nodes.qclosure(mc=100, repeat=5, rng=rng)
        >>> err = jnp.linalg.norm(qc0 - qc1) / jnp.linalg.norm(qc0)
        >>> (err < 0.05).item()
        True
        >>> cor = jnp.corrcoef(qc0, qc1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        return _m1(self)


@eqx.filter_jit
def _m1(stat: RandomGraphQClosure) -> Reals:
    """Compute the first moment of the statistic."""
    kw1, kw2 = stat.split_options(2, repeat=1, average=True)
    quadrangles = stat.nodes.motifs.quadrangle(**kw1)
    qheads = stat.nodes.motifs.qhead(**kw2)
    return stat.m1_from_motifs(quadrangles, qheads)
