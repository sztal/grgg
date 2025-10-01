from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.statistics.motifs import TriangleMotifStatistic
from grgg.utils.compute import MapReduce


class UndirectedRandomGraphTriangleMotifStatistic(TriangleMotifStatistic):
    """Triangle motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Triangle count implementation for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**3 / 2

    def _heterogeneous_m1(
        self,
        *,
        batch_size: int | None = None,
        **kwargs: Any,  # noqa
    ) -> Reals:
        """Triangle count implementation for heterogeneous undirected random graphs."""
        batch_size = self.model._get_batch_size(batch_size)
        n = self.model.n_nodes
        vids = jnp.arange(n)
        use_sampling, key, mr_kwargs, _ = self.prepare_compute_kwargs(**kwargs)

        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Real:
            """Sum over k of p_ik * p_jk."""
            p_ik = self.model.pairs[i].probs()
            p_jk = self.model.pairs[j].probs()
            return jnp.sum(p_ik * p_jk)

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Sum over j of p_ij * sum_k(p_ik * p_jk)."""
            key_k = jax.random.fold_in(key, i) if use_sampling else None

            @MapReduce(rng=key_k, **mr_kwargs)
            def compute(j: Integer) -> Real:
                return jax.lax.cond(
                    j == i,
                    lambda: 0.0,
                    lambda: (self.model.pairs[i, j].probs() * sum_k(i, j)),
                )

            return compute(vids)

        indices = self.nodes.coords[0].flatten()
        triangles = jax.vmap(sum_j)(indices)
        return triangles / 2
