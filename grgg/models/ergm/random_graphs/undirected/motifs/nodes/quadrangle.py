from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import QuadrangleMotif
from grgg.utils.compute import MapReduce


class UndirectedRandomGraphQuadrangleMotif(QuadrangleMotif):
    """Quadrangle motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Quadrangle count for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**4 * (1 - p) ** 2 / 2

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Quadrangle count for heterogeneous undirected random graphs."""
        n = self.model.n_nodes
        vids = jnp.arange(n)
        rng, mr_kwargs, loop_kwargs = self.prepare_compute_kwargs(**kwargs)
        weights = self.importance_weights
        # Computations with inner loops must pass the explicit key
        # other wise jax gets confused
        key = rng.key if rng is not None else None

        @jax.jit
        def sum_l(i: Integer, j: Integer, k: Integer) -> Real:
            """Sum over l of p_kl * p_il * (1 - p_jl)."""
            prob = (
                self.pairs[jnp.array([k, i, j])]
                .probs()
                .at[-1]
                .mul(-1.0)
                .at[-1]
                .add(1.0)
                .prod(axis=0)
                .sum()
            )
            p_ij = self.pairs[i, j].probs()
            p_jk = self.pairs[j, k].probs()
            return prob - p_ij * p_jk

        @jax.jit
        def sum_k(i: Integer, j: Integer, key: Integers | None) -> Real:
            """Sum over k of p_jk * (1 - p_ik) * sum_l"""
            key_k = jax.random.fold_in(key, j) if key is not None else None
            v = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)
            w = weights[v] if self.use_sampling else None

            @MapReduce(rng=key_k, p=w, **mr_kwargs)
            def compute(k: Integer) -> Real:
                return (
                    self.pairs[j, k].probs()
                    * (1 - self.pairs[i, k].probs())
                    * sum_l(i, j, k)
                )

            return compute(v)

        def sum_j(i: Integer) -> Real:
            """Sum over j of p_ij * sum_k"""
            key_j = jax.random.fold_in(key, i) if self.use_sampling else None
            v = jnp.delete(vids, i, assume_unique_indices=True)
            w = weights[v] if self.use_sampling else None

            @MapReduce(rng=key_j, p=w, **mr_kwargs)
            def compute(j: Integer) -> Real:
                return self.pairs[i, j].probs() * sum_k(i, j, key_j)

            return compute(v)

        indices = self.nodes.coords[0].flatten()
        quadrangles = jax.lax.map(sum_j, indices, **loop_kwargs)

        return quadrangles / 2
