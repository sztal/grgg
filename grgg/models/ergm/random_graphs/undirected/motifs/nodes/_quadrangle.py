from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import QuadrangleMotif
from grgg.utils.compute import foreach, sample


class UndirectedRandomGraphQuadrangleMotif(QuadrangleMotif):
    """Quadrangle motif statistic for undirected random graphs.

    Examples
    --------
    Here we show that the importance sampling estimator for the quadrangle motif is
    effective.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import UndirectedRandomGraph, RandomGenerator
    >>> n = 100
    >>> rng = RandomGenerator(303)
    >>> model = UndirectedRandomGraph(n, mu=rng.normal(n) - 3)
    >>> Q_exact = model.nodes.motifs.quadrangle()
    >>> Q_approx = model.nodes.motifs.quadrangle(n_samples=10, rng=rng)
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
        """Quadrangle count for homogeneous undirected random graphs."""
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**4 * (1 - p) ** 2 / 2

    def _heterogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Quadrangle count for heterogeneous undirected random graphs."""
        return _heterogeneous_m1(self, **kwargs)


@eqx.filter_jit
def _heterogeneous_m1(
    stat: UndirectedRandomGraphQuadrangleMotif, **kwargs: Any
) -> Reals:
    """Quadrangle count for heterogeneous undirected random graphs."""
    n = stat.model.n_nodes
    vids = jnp.arange(n)
    key, sample_kwargs, loop_kwargs = stat.prepare_compute_kwargs(**kwargs)
    weights = stat.importance_weights

    @jax.checkpoint
    @jax.jit
    def sum_l(i: Integer, j: Integer, k: Integer) -> Real:
        """Sum over l of p_kl * p_il * (1 - p_jl)."""
        l = jnp.delete(vids, jnp.array([i, j, k]), assume_unique_indices=True)
        probs = stat.pairs[jnp.ix_(jnp.array([k, i]), l)].probs().prod(0)
        comps = 1 - stat.pairs[j, l].probs()
        return (probs * comps).sum(-1)

    @jax.jit
    def sum_k(i: Integer, j: Integer, key: Integers | None) -> Real:
        """Sum over k of p_jk * (1 - p_ik) * sum_l"""
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
            out = (
                stat.pairs[j, k].probs()
                * (1 - stat.pairs[i, k].probs())
                * sum_l(i, j, k)
            )
            if stat.use_sampling:
                out *= x[1]  # importance weight
            return carry + out, None

        return _sum_k[0]  # type: ignore

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
            out = stat.pairs[i, j].probs() * sum_k(i, j, key_j)
            if stat.use_sampling:
                out *= x[1]  # importance weight
            return carry + out, None

        return _sum_j[0]  # type: ignore

    indices = stat.nodes.coords[0]
    quadrangles = jax.lax.map(sum_j, indices, **loop_kwargs)

    return quadrangles / 2
