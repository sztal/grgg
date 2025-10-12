from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.statistics.motifs import TWedgeMotif


class UndirectedRandomGraphTWedgeMotif(TWedgeMotif):
    """T-wedge motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Triangle wedge path count for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**2

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Triangle wedge path count for heterogeneous undirected random graphs."""
        *_, loop_kwargs = self.prepare_compute_kwargs(**kwargs)
        degree = self.nodes.reset().degree()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Sum over j of p_ij * sum_k p_ik."""
            p_ij = self.model.pairs[i].probs()
            return jnp.sum(p_ij * (degree[i] - p_ij))

        indices = self.nodes.coords[0].flatten()
        twedges = jax.lax.map(sum_j, indices, **loop_kwargs)
        return twedges
