from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import QWedgeMotifStatistic
from grgg.utils.compute import MapReduce


class UndirectedRandomGraphQWedgeMotifStatistic(QWedgeMotifStatistic):
    """Q-wedge motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Q-wedge count implementation for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**3

    def _heterogeneous_m1(
        self,
        *,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Reals:
        """Q-wedge count implementation for heterogeneous undirected random graphs."""
        batch_size = self.model._get_batch_size(batch_size)
        n = self.model.n_nodes
        vids = jnp.arange(n)
        use_sampling, key, mr_kwargs, _ = self.prepare_compute_kwargs(**kwargs)

        @jax.jit
        def sum_l(i: Integer, j: Integer, k: Integer) -> Real:
            """Sum over l of p_il."""
            p_ij = self.model.pairs[i, j].probs()
            p_ik = self.model.pairs[i, k].probs()
            return self.model.pairs[i].probs().sum() - p_ij - p_ik

        @jax.jit
        def sum_k(i: Integer, j: Integer, key: Integers | None) -> Real:
            """Sum over k of p_jk * sum_l"""
            key_k = jax.random.fold_in(key, j) if use_sampling else None

            @MapReduce(rng=key_k, **mr_kwargs)
            def compute(k: Integer) -> Real:
                return jax.lax.cond(
                    (k == i) | (k == j),
                    lambda: 0.0,
                    lambda: self.model.pairs[j, k].probs() * sum_l(i, j, k),
                )

            return compute(vids)

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Sum over j of p_ij * sum_k"""
            key_j = jax.random.fold_in(key, i) if use_sampling else None

            @MapReduce(rng=key_j, **mr_kwargs)
            def compute(j: Integer) -> Real:
                return jax.lax.cond(
                    j == i,
                    lambda: 0.0,
                    lambda: self.model.pairs[i, j].probs() * sum_k(i, j, key_j),
                )

            return compute(vids)

        indices = self.nodes.coords[0].flatten()
        qwedges = jax.lax.map(sum_j, indices, batch_size=batch_size)
        return qwedges
