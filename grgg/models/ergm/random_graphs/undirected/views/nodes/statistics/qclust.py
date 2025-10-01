from typing import Any

from grgg._typing import Reals
from grgg.statistics import QClusteringStatistic


class UndirectedRandomGraphQClusteringStatistic(QClusteringStatistic):
    """Degree statistic for undirected random graphs."""

    def _m1(self, **kwargs: Any) -> Reals:
        """Compute the first moment of the statistic."""
        kw1, kw2 = self.split_compute_kwargs(**kwargs)
        quadrangles = self.nodes.motifs.quadrangle(**kw1)
        qwedges = self.nodes.motifs.qwedge(**kw2)
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
        return self._m1(**kwargs)

    def _heterogeneous_m1(
        self,
        *,
        batch_size: int | None = None,
        **kwargs: Any,  # noqa
    ) -> Reals:
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
        return self._m1(batch_size=batch_size, **kwargs)
