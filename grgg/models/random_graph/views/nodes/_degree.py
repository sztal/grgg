from typing import TYPE_CHECKING, Any

from grgg._typing import Integer, Real, Reals
from grgg.statistics import Degree

if TYPE_CHECKING:
    from grgg.models.random_graph import RandomGraph

__all__ = ("RandomGraphDegree",)


class RandomGraphDegree(Degree):
    """Node degree statistic for random graph models.

    Examples
    --------
    Expected degree for homogeneous undirected random graph models.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> model = RandomGraph(100, mu=-2)
    >>> kbar = model.nodes.degree()
    >>> kbar.shape
    ()
    >>> kbar.item()
    11.801089
    >>> K = jnp.array(
    ...     [model.sample(rng=rng).A.sum(axis=1).mean() for _ in range(20)]
    ... )
    >>> jnp.isclose(K.mean(), kbar, rtol=1e-1).item()
    True
    >>> K = model.nodes[...].degree()
    >>> K.shape
    (100,)
    >>> jnp.allclose(K, kbar).item()
    True

    Expected degree for heterogeneous undirected random graph models.
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(42)
    >>> mu = rng.normal(100)
    >>> model = RandomGraph(mu.size, mu=mu)
    >>> D = model.nodes.degree()
    >>> D.shape
    (100,)
    >>> K = jnp.column_stack(
    ...     [model.sample(rng=rng).A.sum(axis=1) for _ in range(20)
    ... ]).mean(axis=1)
    >>> jnp.allclose(D, K, rtol=1e-1).item()
    True
    >>> vids = jnp.array([0, 11, 27, 89])
    >>> D = model.nodes[vids].degree()
    >>> D.shape
    (4,)
    >>> jnp.allclose(D, K[vids], rtol=1e-1).item()
    True

    Check gradients via autodiff.
    >>> def degsum(model): return model.nodes.degree().sum()
    >>> def degsum_naive(model):
    ...     return model.pairs.probs().sum(axis=1).sum()
    >>> grad = jax.grad(degsum)(model)
    >>> grad_naive = jax.grad(degsum_naive)(model)
    >>> jnp.allclose(grad.mu.data, grad_naive.mu.data).item()
    True

    Gradient of sum of squared deviations from target.
    >>> target = rng.normal(100) * 5 + 10
    >>> def loss(model): return ((model.nodes.degree() - target) ** 2).sum()
    >>> grad = jax.grad(loss)(model)
    >>> grad.mu.data.shape
    (100,)
    >>> jnp.isfinite(grad.mu.data).all().item()
    True
    """

    @staticmethod
    def kernel_m1_exact(model: "RandomGraph", i: Integer) -> Real:
        return model.pairs[i].probs().sum()

    def _homogeneous_m1(self, **kwargs: Any) -> Reals:  # noqa
        return self.model.pairs.probs() * (self.model.n_nodes - 1)

    def _heterogeneous_m1_exact(self) -> Reals:  # noqa
        return self.iteration(order=0, kernel=self.kernel_m1_exact).map(
            self.model, self.nodes.coords[0]
        )
