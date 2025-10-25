from typing import Any

import jax
import jax.numpy as jnp

from grgg._typing import Integer, IntVector, Real, Reals, RealVector
from grgg.statistics.motifs import QWedgeMotif
from grgg.utils.compute import foreach


class RandomGraphQWedgeMotif(QWedgeMotif):
    """Q-wedge motif statistic for undirected random graphs."""

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        """Q-wedge count implementation for homogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
        >>> qwedge = model.nodes.motifs.qwedge()
        >>> QW = jnp.array(
        ...     [model.sample(rng=rng).struct.census().qw.mean() for _ in range(20)]
        ... )
        >>> jnp.isclose(qwedge, QW.mean(), rtol=5e-2).item()
        True
        """
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**3

    def _heterogeneous_m1_exact(self) -> Reals:
        """Q-wedge count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
        >>> qwedge = model.nodes.motifs.qwedge()
        >>> QW = jnp.column_stack(
        ...     [model.sample(rng=rng).struct.census().qw.to_numpy() for _ in range(20)]
        ... ).mean(axis=-1)
        >>> jnp.isclose(qwedge.mean(), QW.mean(), rtol=1e-1).item()
        True
        >>> (jnp.corrcoef(qwedge, QW)[0, 1] > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)
        degree = self.model.nodes.degree()

        @jax.checkpoint
        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Real:
            """Sum over k of p_jk * (d_i - p_ij - p_ik)."""
            return self._inner_sum(degree, vids, i, j)

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Expected sum over k of p_jk * (d_i - p_ij - p_ik)."""
            j = jnp.delete(vids, i, assume_unique_indices=True)

            @foreach(j, init=0.0, unroll=self.unroll)
            def expectation(carry: Real, j: Integer) -> tuple[Real, None]:
                p_ij = self.model.pairs[i, j].probs()
                return carry + p_ij * sum_k(i, j), None

            return expectation[0]  # type: ignore

        indices = self.nodes.coords[0]
        qwedges = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return qwedges

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo q-wedge count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> qw0 = jnp.log(model.nodes.motifs.qwedge())
        >>> qw1 = jnp.log(model.nodes.motifs.qwedge(mc=50, rng=rng))
        >>> err = jnp.linalg.norm(qw0 - qw1) / jnp.linalg.norm(qw0)
        >>> (err < 0.05).item()
        True
        >>> cor = jnp.corrcoef(qw0, qw1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)
        degree = self.model.nodes.degree()

        @jax.checkpoint
        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Real:
            """Sum over k of p_jk * (d_i - p_ij - p_ik)."""
            return self._inner_sum(degree, vids, i, j)

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Expected sum over k of p_jk * (d_i - p_ij - p_ik)."""
            key = jax.random.fold_in(self.key, i)
            j = jnp.delete(vids, i, assume_unique_indices=True)
            p_ij = self.model.pairs[i, j].probs()
            d_i = degree[i]
            j = jax.random.choice(key, j, (self.mc,), replace=True, p=p_ij)

            @foreach(j, init=0.0, unroll=self.unroll)
            def expectation(carry: Real, j: Integer) -> tuple[Real, None]:
                return carry + sum_k(i, j), None

            return d_i / self.mc * expectation[0]  # type: ignore

        indices = self.nodes.coords[0]
        qwedges = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return qwedges

    def _inner_sum(
        self,
        degree: RealVector,
        vids: IntVector,
        i: Integer,
        j: Integer,
    ) -> Real:
        k = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)
        p_jk = self.model.pairs[j, k].probs()
        d_i = degree[i]
        p_ij = self.model.pairs[i, j].probs()
        p_ik = self.model.pairs[i, k].probs()
        return jnp.sum(p_jk * (d_i - p_ij - p_ik))
