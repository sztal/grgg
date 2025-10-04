from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.statistics.motifs import TriangleMotif
from grgg.utils.compute import MapReduce


class UndirectedRandomGraphTriangleMotif(TriangleMotif):
    """Triangle motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Triangle count implementation for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**3 / 2

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:
        """Triangle count implementation for heterogeneous undirected random graphs."""
        n = self.model.n_nodes
        vids = jnp.arange(n)
        rng, mr_kwargs, loop_kwargs = self.prepare_compute_kwargs(**kwargs)
        weights = self.importance_weights
        # Computations with inner loops must pass the explicit key
        # other wise jax gets confused
        key = rng.key if rng is not None else None

        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Real:
            """Sum over k of p_ik * p_jk."""
            p_ik = self.model.pairs[i].probs()
            p_jk = self.model.pairs[j].probs()
            return jnp.sum(p_ik * p_jk)

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Sum over j of p_ij * sum_k(p_ik * p_jk)."""
            key_j = jax.random.fold_in(key, i) if self.use_sampling else None
            v = jnp.delete(vids, i, assume_unique_indices=True)
            w = weights[v] if self.use_sampling else None

            @MapReduce(rng=key_j, p=w, **mr_kwargs)
            def compute(j: Integer) -> Real:
                return self.model.pairs[i, j].probs() * sum_k(i, j)

            return compute(v)

        indices = self.nodes.coords[0].flatten()
        triangles = jax.lax.map(sum_j, indices, **loop_kwargs)
        return triangles / 2
