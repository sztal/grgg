from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.statistics.motifs import TWedgeMotifStatistic


class UndirectedRandomGraphTWedgeMotifStatistic(TWedgeMotifStatistic):
    """T-wedge motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Triangle wedge path count for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**2

    def _heterogeneous_m1(
        self,
        *,
        batch_size: int | None = None,
        **kwargs: Any,  # noqa
    ) -> Reals:
        """Triangle wedge path count for heterogeneous undirected random graphs."""
        batch_size = self.model._get_batch_size(batch_size)
        degree = self.nodes.reset().degree()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Sum over j of p_ij * sum_k p_ik."""
            p_ij = self.model.pairs[i].probs()
            return jnp.sum(p_ij * (degree[i] - p_ij))

        indices = self.nodes.coords[0].flatten()
        twedges = jax.vmap(sum_j)(indices)
        return twedges
