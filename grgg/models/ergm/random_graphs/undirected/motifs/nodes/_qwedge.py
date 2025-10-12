from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import QWedgeMotif
from grgg.utils.compute import foreach, sample


class UndirectedRandomGraphQWedgeMotif(QWedgeMotif):
    """Q-wedge motif statistic for undirected random graphs.

    Examples
    --------
    Here we show that the importance sampling estimator for the Q-wedge motif is
    effective.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import UndirectedRandomGraph, RandomGenerator
    >>> n = 100
    >>> rng = RandomGenerator(303)
    >>> model = UndirectedRandomGraph(n, mu=rng.normal(n) - 3)
    >>> Q_exact = model.nodes.motifs.qwedge()
    >>> Q_approx = model.nodes.motifs.qwedge(n_samples=10, rng=rng)
    >>>
    >>> def error(X, Y):
    ...     return jnp.linalg.norm(X - Y) / jnp.linalg.norm(X)
    >>>
    >>> err = error(Q_exact, Q_approx)
    >>> (err < 0.25).item()
    True
    >>> cor = jnp.corrcoef(Q_exact, Q_approx)[0, 1]
    >>> (cor > 0.95).item()
    True
    """

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Q-wedge count implementation for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**3

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Q-wedge count for heterogeneous undirected random graphs."""
        return _heterogeneous_m1(self, **kwargs)


@eqx.filter_jit
def _heterogeneous_m1(stat: UndirectedRandomGraphQWedgeMotif, **kwargs: Any) -> Reals:
    """Q-wedge count implementation for heterogeneous undirected random graphs."""
    n = stat.model.n_nodes
    vids = jnp.arange(n)
    key, sample_kwargs, loop_kwargs = stat.prepare_compute_kwargs(**kwargs)
    weights = stat.importance_weights

    @jax.checkpoint
    @jax.jit
    def sum_l(i: Integer, j: Integer, k: Integer) -> Real:
        """Sum over l of p_il."""
        p_ij_ik = stat.model.pairs[[i, i], [j, k]].probs().sum(-1)
        return stat.model.pairs[i].probs().sum(-1) - p_ij_ik

    @jax.jit
    def sum_k(i: Integer, j: Integer, key: Integers | None) -> Real:
        """Sum over k of p_jk * sum_l"""
        key_k = jax.random.fold_in(key, j) if key is not None else None
        v = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)
        xs = (
            sample(v, p=weights[v], rng=key_k, **sample_kwargs)
            if stat.use_sampling
            else (v,)
        )

        @foreach(xs, init=0.0)
        def _sum_k(carry: Real, x: Integer | tuple[Integer, Real]) -> tuple[Real, None]:
            k = x[0]
            out = stat.model.pairs[j, k].probs() * sum_l(i, j, k)
            if stat.use_sampling:
                out *= x[1]  # importance weight
            return carry + out, None

        return _sum_k[0]  # type: ignore

    @jax.jit
    def sum_j(i: Integer) -> Real:
        """Sum over j of p_ij * sum_k"""
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
            out = stat.model.pairs[i, j].probs() * sum_k(i, j, key_j)
            if stat.use_sampling:
                out *= x[1]  # importance weight
            return carry + out, None

        return _sum_j[0]  # type: ignore

    indices = stat.nodes.coords[0]
    qwedges = jax.lax.map(sum_j, indices, **loop_kwargs)
    return qwedges
