import jax
import jax.numpy as jnp

from grgg._typing import Integer, IntVector, Real, Reals
from grgg.statistics.motifs import TriangleMotif
from grgg.utils.compute import foreach


class RandomGraphTriangleMotif(TriangleMotif):
    """Triangle motif statistic for undirected random graphs."""

    def _homogeneous_m1(self) -> Reals:
        """Triangle count implementation for homogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
        >>> triangle = model.nodes.motifs.triangle()
        >>> T = jnp.array(
        ...     [model.sample(rng=rng).struct.census().t.mean() for _ in range(20)]
        ... )
        >>> jnp.isclose(triangle, T.mean(), rtol=5e-2).item()
        True
        """
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**3 / 2

    def _heterogeneous_m1_exact(self) -> Reals:  # noqa
        """Triangle count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
        >>> triangle = model.nodes.motifs.triangle()
        >>> T = jnp.column_stack(
        ...     [model.sample(rng=rng).struct.census().t.to_numpy() for _ in range(20)]
        ... ).mean(axis=-1)
        >>> jnp.isclose(triangle.mean(), T.mean(), rtol=1e-2).item()
        True
        >>> (jnp.corrcoef(triangle, T)[0, 1] > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)

        @jax.checkpoint
        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Real:
            """Sum over k of p_ik * p_jk."""
            k = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)
            return self._inner(i, j, k).sum()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Expected sum over k of p_ik * p_jk."""
            j = jnp.delete(vids, i, assume_unique_indices=True)

            @foreach(j, init=0.0, unroll=self.unroll)
            def expectation(carry: Real, j: Integer) -> tuple[Real, None]:
                p_ij = self.model.pairs[i, j].probs()
                return carry + p_ij * sum_k(i, j), None

            return expectation[0]  # type: ignore

        indices = self.nodes.coords[0]
        triangles = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return triangles / 2

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo triangle count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 500
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> t0 = jnp.log(model.nodes.motifs.triangle())
        >>> t1 = jnp.log(model.nodes.motifs.triangle(mc=100, rng=rng))
        >>> err = jnp.linalg.norm(t0 - t1) / jnp.linalg.norm(t0)
        >>> (err < 0.05).item()
        True
        >>> cor = jnp.corrcoef(t0, t1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)

        @jax.checkpoint
        @jax.jit
        def sum_k(i: Integer, j: Integer) -> Real:  # noqa
            """Sum over k of p_ik * p_jk."""
            k = jnp.delete(vids, jnp.array([i, j]), assume_unique_indices=True)
            return self._inner(i, j, k).sum()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            """Expected sum over k of p_ik * p_jk."""
            key = jax.random.fold_in(self.key, i)
            j = jnp.delete(vids, i, assume_unique_indices=True)
            p_ij = self.model.pairs[i, j].probs()
            j = jax.random.choice(key, j, (self.mc,), replace=True, p=p_ij)
            d_i = p_ij.sum()

            @foreach(j, init=0.0, unroll=self.unroll)
            def expectation(carry: Real, j: Integer) -> tuple[Real, None]:
                return carry + sum_k(i, j), None

            return d_i / self.mc * expectation[0]  # type: ignore

        indices = self.nodes.coords[0]
        triangles = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return triangles / 2

    def _inner(self, i: Integer, j: Integer, k: IntVector) -> Real:
        ij = jnp.array([i, j])
        return self.model.pairs[jnp.ix_(ij, k)].probs().prod(0)
