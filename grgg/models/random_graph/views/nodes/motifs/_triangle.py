from typing import TYPE_CHECKING

import jax.numpy as jnp

from grgg._typing import Integer, Integers, Real, Reals
from grgg.statistics.motifs import TriangleMotif

if TYPE_CHECKING:
    from grgg.models.random_graph import RandomGraph

__all__ = ("RandomGraphTriangleMotif",)


class RandomGraphTriangleMotif(TriangleMotif):
    """Triangle motif statistic for undirected random graphs.

    Examples
    --------
    Homogeneous case.
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

    Heterogeneous case (exact).
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

    Heterogeneous case (Monte Carlo).
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

    Gradient of sum of squared deviations from target.
    >>> import jax
    >>> model = RandomGraph(50, mu=rng.normal(50) - 1.5)
    >>> target = rng.exponential(50) * 10
    >>> def loss(model): return ((model.nodes.motifs.triangle() - target) ** 2).sum()
    >>> grad = jax.grad(loss)(model)
    >>> grad.mu.data.shape
    (50,)
    >>> jnp.isfinite(grad.mu.data).all().item()
    True
    """

    @staticmethod
    def _kernel_m1_exact(model: "RandomGraph", i: Integer, j: Integer) -> Real:
        p_ij = model.pairs[i, j].probs()
        ij = jnp.array([i, j])
        k = jnp.arange(model.n_nodes)
        t = p_ij * model.pairs[jnp.ix_(ij, k)].probs().prod(0).sum()
        return t / 2

    @staticmethod
    def _kernel_m1_mc(model: "RandomGraph", i: Integer, j: Integer) -> Real:
        ij = jnp.array([i, j])
        k = jnp.arange(model.n_nodes)
        t = model.pairs[jnp.ix_(ij, k)].probs().prod(0).sum()
        return t / 2

    @staticmethod
    def _mc_weights(model: "RandomGraph", depth: int, vids: Integers) -> Reals:
        if depth == 0:
            return model.pairs[vids[depth]].probs()
        raise NotImplementedError

    def _homogeneous_m1(self) -> Reals:
        n = self.model.n_nodes
        p = self.model.pairs.probs()
        return (n - 1) * (n - 2) * p**3 / 2

    def _heterogeneous_m1_exact(self) -> Reals:
        return self.iteration(
            order=1,
            kernel=self._kernel_m1_exact,
            unique=False,
        ).map(self.model, self.nodes.coords[0])

    def _heterogeneous_m1_monte_carlo(self) -> Reals:
        return self.iteration(
            order=1,
            kernel=self._kernel_m1_mc,
            weights=self._mc_weights,
            unique=True,
        ).map(self.model, self.nodes.coords[0])
