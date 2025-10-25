import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals, RealVector
from grgg.statistics.motifs import TWedgeMotif


class RandomGraphTWedgeMotif(TWedgeMotif):
    """T-wedge motif statistic for undirected random graphs."""

    def _homogeneous_m1(self) -> Reals:
        """Triangle wedge path count for homogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
        >>> twedge = model.nodes.motifs.twedge()
        >>> TW = jnp.array(
        ...     [model.sample(rng=rng).struct.census().tw.mean() for _ in range(20)]
        ... )
        >>> jnp.isclose(twedge, TW.mean(), rtol=5e-2).item()
        True
        """
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**2

    def _heterogeneous_m1_exact(self) -> Reals:
        """Triangle wedge path count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
        >>> twedge = model.nodes.motifs.twedge()
        >>> TW = jnp.column_stack(
        ...     [model.sample(rng=rng).struct.census().tw.to_numpy() for _ in range(20)]
        ... ).mean(axis=-1)
        >>> jnp.isclose(twedge.mean(), TW.mean(), rtol=1e-1).item()
        True
        >>> (jnp.corrcoef(twedge, TW)[0, 1] > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)
        degree = self.model.nodes.degree()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            j = jnp.delete(vids, i, assume_unique_indices=True)
            d_i = degree[i]
            p_ij = self.model.pairs[i, j].probs()
            return jnp.sum(p_ij * (d_i - p_ij))

        indices = self.nodes.coords[0]
        twedges = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return twedges

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo t-wedge count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> tw0 = jnp.log(model.nodes.motifs.twedge())
        >>> tw1 = jnp.log(model.nodes.motifs.twedge(mc=100, rng=rng))
        >>> err = jnp.linalg.norm(tw0 - tw1) / jnp.linalg.norm(tw0)
        >>> (err < 0.01).item()
        True
        >>> cor = jnp.corrcoef(tw0, tw1)[0, 1]
        >>> (cor > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)
        degree = self.model.nodes.degree()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            key = jax.random.fold_in(self.key, i)
            j = jnp.delete(vids, i, assume_unique_indices=True)
            p_ij = self.model.pairs[i, j].probs()
            indices = jax.random.choice(key, j, (self.mc,), replace=True, p=p_ij)
            d_i = degree[i]
            return d_i / self.mc * jnp.sum(d_i - p_ij[indices])

        indices = self.nodes.coords[0]
        twedges = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return twedges

    def _inner_sum(self, degree: RealVector, i: Integer, j: Integer) -> Real:
        return degree[i] - self.model.pairs[i, j].probs()
