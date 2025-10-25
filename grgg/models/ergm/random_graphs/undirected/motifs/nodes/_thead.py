import jax
import jax.numpy as jnp

from grgg._typing import Integer, Real, Reals, RealVector
from grgg.statistics.motifs import THeadMotif


class RandomGraphTHeadMotif(THeadMotif):
    """T-head motif statistic for undirected random graphs."""

    def _homogeneous_m1(self) -> Reals:
        """Triangle head path count for homogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> model = RandomGraph(100, mu=-2)
        >>> thead = model.nodes.motifs.thead()
        >>> TH = jnp.array(
        ...     [model.sample(rng=rng).struct.census().th.mean() for _ in range(20)]
        ... )
        >>> jnp.isclose(thead, TH.mean(), rtol=5e-2).item()
        True
        """
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**2

    def _heterogeneous_m1_exact(self) -> Reals:
        """Triangle head path count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> model = RandomGraph(100, mu=rng.normal(100) - 1.2)
        >>> thead = model.nodes.motifs.thead()
        >>> TH = jnp.column_stack(
        ...     [model.sample(rng=rng).struct.census().th.to_numpy() for _ in range(20)]
        ... ).mean(axis=-1)
        >>> jnp.isclose(thead.mean(), TH.mean(), rtol=1e-1).item()
        True
        >>> (jnp.corrcoef(thead, TH)[0, 1] > 0.99).item()
        True
        """
        vids = jnp.arange(self.model.n_nodes)
        degree = self.model.nodes.degree()

        @jax.jit
        def sum_j(i: Integer) -> Real:
            j = jnp.delete(vids, i, assume_unique_indices=True)
            d_j = degree[j]
            p_ij = self.model.pairs[i, j].probs()
            return jnp.sum(p_ij * (d_j - p_ij))

        indices = self.nodes.coords[0]
        theads = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return theads

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        """Monte Carlo t-head count for heterogeneous undirected random graphs.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(42)
        >>> n = 1000
        >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
        >>> th0 = jnp.log(model.nodes.motifs.thead())
        >>> th1 = jnp.log(model.nodes.motifs.thead(mc=100, rng=rng))
        >>> err = jnp.linalg.norm(th1 - th0) / jnp.linalg.norm(th0)
        >>> (err < 0.02).item()
        True
        >>> cor = jnp.corrcoef(th0, th1)[0, 1]
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
            indices = jax.random.choice(key, len(j), (self.mc,), replace=True, p=p_ij)
            p_ij = p_ij[indices]
            d_j = degree[j[indices]]
            d_i = degree[i]
            return d_i / self.mc * jnp.sum(d_j - p_ij)

        indices = self.nodes.coords[0]
        theads = jax.lax.map(sum_j, indices, batch_size=self.batch_size)
        return theads

    def _inner_sum(self, degree: RealVector, i: Integer, j: Integer) -> Real:
        return degree[j] - self.model.pairs[i, j].probs()
