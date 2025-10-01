from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import QHeadMotifStatistic
from grgg.utils.compute import MapReduce


class UndirectedRandomGraphQHeadMotifStatistic(QHeadMotifStatistic):
    """Q-head motif statistic for undirected random graphs.

    Examples
    --------
    Here we show that the importance sampling estimator for the Q-head motif is
    unbiased and very effective.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import UndirectedRandomGraph, RandomGenerator
    >>> n = 100
    >>> rng = RandomGenerator(303)
    >>> model = UndirectedRandomGraph(n, mu=rng.normal(n) - 3)
    >>> Q_exact = model.nodes.motifs.qhead()
    >>> Q_approx = model.nodes.motifs.qhead(n_samples=10, rng=rng)
    """

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Q-head count implementation for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**3

    def _heterogeneous_m1(
        self,
        *,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Reals:
        """Q-head count implementation for heterogeneous undirected random graphs."""
        batch_size = self.model._get_batch_size(batch_size)
        n = self.model.n_nodes
        vids = jnp.arange(n)
        use_sampling, key, mr_kwargs, _ = self.prepare_compute_kwargs(**kwargs)

        @jax.jit
        def sum_l(i: Integer, j: Integer, k: Integer) -> Real:
            """Sum over l of p_kl."""
            p_ik = self.model.pairs[i, k].probs()
            p_jk = self.model.pairs[j, k].probs()
            return self.model.pairs[k].probs().sum() - p_ik - p_jk

        @jax.jit
        def sum_k(i: Integer, j: Integer, key: Integers | None) -> Real:
            """Sum over k of p_ik * sum_l"""
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
                    lambda: self.model.pairs[i, j].probs() * sum_k(i, j, key=key_j),
                )

            return compute(vids)

        indices = self.view.coords[0].flatten()
        qheads = jax.lax.map(sum_j, indices, batch_size=batch_size)
        return qheads
