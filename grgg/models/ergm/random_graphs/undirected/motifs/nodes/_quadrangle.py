import jax
import jax.numpy as jnp

from grgg._typing import Integer, Integers, IntVector, Real, Reals
from grgg.statistics.motifs import QuadrangleMotif
from grgg.utils.compute import foreach


class RandomGraphQuadrangleMotif(QuadrangleMotif):
    """Quadrangle motif statistic for undirected random graphs."""

    def _homogeneous_m1(self) -> Reals:
        """Quadrangle count for homogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
        >>> quadrangle = model.nodes.motifs.quadrangle()
        >>> Q = jnp.array(
        ...     [model.sample(rng=rng).struct.census().q0.mean() for _ in range(20)]
        ... )
        >>> jnp.isclose(quadrangle, Q.mean(), rtol=5e-2).item()
        True
        """
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**4 * (1 - p) ** 2 / 2

    def _heterogeneous_m1_exact(self) -> Reals:
        """Quadrangle count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
        >>> quadrangle = model.nodes.motifs.quadrangle()
        >>> Q = jnp.column_stack(
        ...     [model.sample(rng=rng).struct.census().q0.to_numpy() for _ in range(20)]
        ... ).mean(axis=-1)
        >>> jnp.isclose(quadrangle.mean(), Q.mean(), rtol=1e-1).item()
        True
        >>> (jnp.corrcoef(quadrangle, Q)[0, 1] > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)

        @jax.checkpoint
        @jax.jit
        def sum_l(i: Integer, j: Integer, k: Integer) -> Reals:
            return self._inner_sum(vids, i, j, k)

        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Reals:
            k = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)

            @foreach(k, init=0.0, unroll=self.unroll)
            def summation(carry: Real, k: Integer) -> tuple[Real, None]:
                p_jk = self.model.pairs[j, k].probs()
                p_ik = self.model.pairs[i, k].probs()
                return carry + p_jk * (1 - p_ik) * sum_l(i, j, k), None

            return summation[0]  # type: ignore

        @jax.jit
        def sum_j(i: Integer) -> Reals:
            j = jnp.delete(vids, i, assume_unique_indices=True)

            @foreach(j, init=0.0, unroll=self.unroll)
            def summation(carry: Real, j: Integer) -> tuple[Real, None]:
                p_ij = self.model.pairs[i, j].probs()
                return carry + p_ij * sum_k(i, j), None

            return summation[0]  # type: ignore

        indices = self.nodes.coords[0]
        quadrangles = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return quadrangles / 2

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Quadrangle count estimator for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> q0 = jnp.log(model.nodes.motifs.quadrangle())
        >>> q1 = jnp.log(model.nodes.motifs.quadrangle(mc=50, rng=rng))
        >>> err = jnp.linalg.norm(q0 - q1) / jnp.linalg.norm(q0)
        >>> (err < 0.05).item()
        True
        >>> cor = jnp.corrcoef(q0, q1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)
        degree = self.model.nodes.degree()
        indices = self.nodes.coords[0]

        @jax.checkpoint
        @jax.jit
        def sum_l(i: Integer, j: Integer, k: Integer) -> Reals:
            return self._inner_sum(vids, i, j, k)

        @jax.jit
        def sum_k(i: Integer, j: Integer, key: Integers) -> Real:
            k = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)
            p_jk = self.model.pairs[j, k].probs()
            d_j = degree[j]
            p_ij = self.model.pairs[i, j].probs()
            key = jax.random.fold_in(key, j)
            k = jax.random.choice(key, k, (self.mc,), replace=True, p=p_jk)

            @foreach(k, init=0.0, unroll=self.unroll)
            def expectation(carry: Real, k: Integer) -> tuple[Real, None]:
                p_ik = self.model.pairs[i, k].probs()
                return carry + (1 - p_ik) * sum_l(i, j, k), None

            return (d_j - p_ij) / self.mc * expectation[0]  # type: ignore

        @jax.jit
        def sum_j(i: Integer) -> Real:
            d_i = degree[i]
            j = jnp.delete(vids, i, assume_unique_indices=True)
            p_ij = self.model.pairs[i, j].probs()
            key = jax.random.fold_in(self.key, i)
            j = jax.random.choice(key, j, (self.mc,), replace=True, p=p_ij)

            @foreach(j, init=0.0, unroll=self.unroll)
            def expectation(carry: Real, j: Integer) -> tuple[Real, None]:
                return carry + sum_k(i, j, key=key), None

            return d_i / self.mc * expectation[0]  # type: ignore

        quadrangles = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return quadrangles / 2

    def _inner_sum(self, vids: IntVector, i: Integer, j: Integer, k: Integer) -> Real:
        """Sum over l of p_kl * p_il * (1 - p_jl)."""
        l = jnp.delete(vids, jnp.array([i, j, k]), assume_unique_indices=True)
        p_kl = self.model.pairs[k, l].probs()
        p_il = self.model.pairs[i, l].probs()
        p_jl = self.model.pairs[j, l].probs()
        return jnp.sum(p_kl * p_il * (1 - p_jl))
