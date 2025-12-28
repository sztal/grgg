from typing import TYPE_CHECKING

import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import QuadrangleMotif

if TYPE_CHECKING:
    from grgg.models.random_graph import RandomGraph

__all__ = ("RandomGraphQuadrangleMotif",)


class RandomGraphQuadrangleMotif(QuadrangleMotif):
    """Quadrangle motif statistic for undirected random graphs.

    Examples
    --------
    Homogeneous case.
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

    Heterogeneous case (exact).
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

    Heterogeneous case (Monte Carlo).
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> n = 500
    >>> model = RandomGraph(n, mu=rng.normal(n) - 2.5)
    >>> q0 = jnp.log(model.nodes.motifs.quadrangle())
    >>> q1 = jnp.log(model.nodes.motifs.quadrangle(mc=50, rng=rng))
    >>> err = jnp.linalg.norm(q0 - q1) / jnp.linalg.norm(q0)
    >>> (err < 0.05).item()
    True
    >>> cor = jnp.corrcoef(q0, q1)[0, 1]
    >>> (cor > 0.99).item()
    True

    Gradient of sum of squared deviations from target.
    >>> import jax
    >>> model = RandomGraph(30, mu=rng.normal(30) - 1.2)
    >>> target = rng.exponential(30) * 100
    >>> def loss(model): return ((model.nodes.motifs.quadrangle() - target) ** 2).sum()
    >>> grad = jax.grad(loss)(model)
    >>> grad.mu.data.shape
    (30,)
    >>> jnp.isfinite(grad.mu.data).all().item()
    True
    """

    @staticmethod
    def _kernel_m1_exact(
        model: "RandomGraph", i: Integer, j: Integer, k: Integer
    ) -> Real:
        l = jnp.arange(model.n_nodes)
        q = model.pairs[[i, j], [j, k]].probs().prod()
        q = q * model.pairs[jnp.ix_(jnp.array([k, i]), l)].probs().prod(0)
        q = q * (1 - model.pairs[i, k].probs())
        q = q * (1 - model.pairs[j, l].probs())
        return jnp.sum(q.at[jnp.array([i, j, k])].set(0.0)) / 2

    @staticmethod
    def _kernel_m1_mc(model: "RandomGraph", i: Integer, j: Integer, k: Integer) -> Real:
        l = jnp.arange(model.n_nodes)
        ki = jnp.array([k, i])
        q = model.pairs[jnp.ix_(ki, l)].probs().prod(0)
        q = q * (1 - model.pairs[j, l].probs())
        return jnp.sum(q.at[jnp.array([i, j, k])].set(0.0)) / 2

    @staticmethod
    def _mc_weights(model: "RandomGraph", depth: int, vids: Integers) -> Reals:
        if depth == 0:
            return model.pairs[vids[depth]].probs()
        if depth == 1:
            p_jk = model.pairs[vids[1]].probs()
            p_ik = model.pairs[vids[0]].probs()
            return (p_jk * (1 - p_ik)).at[vids[:2]].set(0.0)
        errmsg = f"invalid 'depth={depth}' for quadrangle motif sampling weights"
        raise ValueError(errmsg)

    def _homogeneous_m1(self) -> Reals:
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * (n - 3) * p**4 * (1 - p) ** 2 / 2

    def _heterogeneous_m1_exact(self) -> Reals:
        return self.iteration(
            order=2,
            kernel=self._kernel_m1_exact,
            unique=True,
        ).map(self.model, self.nodes.coords[0])

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        return self.iteration(
            order=2,
            kernel=self._kernel_m1_mc,
            weights=self._mc_weights,
            unique=True,
        ).map(self.model, self.nodes.coords[0])
