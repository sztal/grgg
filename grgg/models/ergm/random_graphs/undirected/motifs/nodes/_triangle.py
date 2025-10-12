from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals
from grgg.statistics.motifs import TriangleMotif
from grgg.utils.compute import foreach, sample


class UndirectedRandomGraphTriangleMotif(TriangleMotif):
    """Triangle motif statistic for undirected random graphs.

    Examples
    --------
    Here we show that the importance sampling estimator for the triangle motif is
    effective.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import UndirectedRandomGraph, RandomGenerator
    >>> n = 100
    >>> rng = RandomGenerator(303)
    >>> model = UndirectedRandomGraph(n, mu=rng.normal(n) - 3)
    >>> T_exact = model.nodes.motifs.triangle()
    >>> T_approx = model.nodes.motifs.triangle(n_samples=10, rng=rng)
    >>>
    >>> def error(X, Y):
    ...     return jnp.linalg.norm(X - Y) / jnp.linalg.norm(X)
    >>>
    >>> err = error(T_exact, T_approx)
    >>> (err < 0.25).item()
    True
    >>> cor = jnp.corrcoef(T_exact, T_approx)[0, 1]
    >>> (cor > 0.95).item()
    True
    """

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Triangle count implementation for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**3 / 2

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Triangle count for heterogeneous undirected random graphs."""
        return _heterogeneous_m1(self, **kwargs)


@eqx.filter_jit
def _heterogeneous_m1(stat: UndirectedRandomGraphTriangleMotif, **kwargs: Any) -> Reals:
    """Triangle count implementation for heterogeneous undirected random graphs."""
    vids = jnp.arange(stat.model.n_nodes)
    key, sample_kwargs, loop_kwargs = stat.prepare_compute_kwargs(**kwargs)
    weights = stat.importance_weights

    @jax.checkpoint
    @jax.jit
    def sum_k(i: Integer, j: Integer) -> Real:
        """Sum over k of p_ik * p_jk."""
        return stat.model.pairs[[i, j]].probs().prod(0).sum(-1)

    @jax.jit
    def sum_j(i: Integer) -> Real:
        """Sum over j of p_ij * sum_k(p_ik * p_jk)."""
        key_j = jax.random.fold_in(key, i) if stat.use_sampling else None
        v = jnp.delete(vids, i, assume_unique_indices=True)
        xs = (
            sample(v, p=weights[v], rng=key_j, **sample_kwargs)
            if stat.use_sampling
            else (v,)
        )

        @foreach(xs, init=0.0)
        def _sum_j(carry: Real, x: Integer | tuple[Integer, Real]) -> tuple[Real, None]:
            j = x[0]
            p_ij = stat.model.pairs[i, j].probs()
            out = p_ij * sum_k(i, j)
            if stat.use_sampling:
                out *= x[1]  # importance weight
            return carry + out, None

        return _sum_j[0]  # type: ignore

    indices = stat.nodes.coords[0]
    triangles = jax.lax.map(sum_j, indices, **loop_kwargs)
    return triangles / 2
