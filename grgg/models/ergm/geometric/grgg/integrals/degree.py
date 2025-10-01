from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg.utils.integrate import IntegrandT
from grgg.utils.misc import number_of_batches

from .abc import AbstractNodesIntegral

if TYPE_CHECKING:
    from grgg.models.geometric.grgg import GRGG

__all__ = ("DegreeIntegral",)

HeterogeneousCarryT = tuple["GRGG", int, jnp.ndarray]
InnerSumT = Callable[[jnp.ndarray], jnp.ndarray]
InnerSumScanT = Callable[
    [HeterogeneousCarryT, ...], tuple[HeterogeneousCarryT, jnp.ndarray]
]


class DegreeIntegral(AbstractNodesIntegral):
    """Abstract base class for degree integrals.

    It is also used to construct concrete degree integrals using
    :meth:`from_model`.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]`.
    """

    @property
    def constant(self) -> float:
        delta = self.model.delta
        R = self.model.manifold.linear_size
        d = self.model.manifold.dim
        dV = self.model.manifold.__class__(d - 1).volume
        return delta * R**d * dV


@DegreeIntegral.register_homogeneous
class HomogeneousDegreeIntegral(DegreeIntegral):
    """Homogeneous degree integral.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from grgg import GRGG, Sphere, Similarity, Complementarity, RandomGenerator
    >>> from grgg.models.grgg.integrals import DegreeIntegral
    >>> rng = RandomGenerator(42)
    >>> model = GRGG(100, 3, Similarity, Complementarity)  # homogeneous
    >>> degree = DegreeIntegral.from_model(model)
    >>> kbar, _ = degree.integrate()
    >>> expected = jnp.asarray(
    ...     [model.nodes.sample_pmatrix(rng=rng).sum(axis=1).mean() for _ in range(50)]
    ... ).mean()
    >>> jnp.isclose(kbar, expected, rtol=1e-2).item()
    True

    Integral is jit-compatible and differentiable.
    >>> @jax.jit
    ... def compute_kbar(model):
    ...     return DegreeIntegral.from_model(model).integrate()[0]
    >>> jnp.isclose(compute_kbar(model), kbar, rtol=1e-5).item()
    True
    >>> grad = jax.grad(compute_kbar)
    >>> grad(model).parameters.array.round(2)
    Array([[15.  , -4.15, 15.  , -4.15]], dtype=...)

    Check that it works for other dimensions and volumes.
    >>> model = GRGG(100, Sphere(4, r=1.0), Similarity, Complementarity)
    >>> degree = DegreeIntegral.from_model(model)
    >>> kbar, _ = degree.integrate()
    >>> expected = jnp.asarray(
    ...     [model.nodes.sample_pmatrix(rng=rng).sum(axis=1).mean() for _ in range(50)]
    ... ).mean()
    >>> jnp.isclose(kbar, expected, rtol=1e-2).item()
    True
    """

    @property
    def constant(self) -> float:
        n = self.model.n_nodes
        return super().constant * (n - 1) / n

    def define_integrand(self) -> IntegrandT:
        @jax.jit
        def integrand(theta: jnp.ndarray) -> jnp.ndarray:
            """Compute the integrand at given angle(s) `theta`."""
            d = self.model.manifold.dim
            R = self.model.manifold.linear_size
            return jnp.sin(theta) ** (d - 1) * self.nodes.pairs.probs(theta * R)

        return integrand


@DegreeIntegral.register_heterogeneous
class HeterogeneousDegreeIntegral(DegreeIntegral):
    """Heterogeneous degree integral.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from grgg import GRGG, Sphere, Similarity, Complementarity, RandomGenerator
    >>> from grgg.models.grgg.integrals import DegreeIntegral
    >>> rng = RandomGenerator(42)
    >>> n = 100
    >>> model = (
    ...     GRGG(n, 2) +
    ...     Similarity(rng.normal(n), rng.normal(n)**2) +
    ...     Complementarity(rng.normal(n), rng.normal(n)**2)
    ... )
    >>> degree = DegreeIntegral.from_model(model)
    >>> kbar, _ = degree.integrate()
    >>> expected = jnp.stack(
    ...     [model.nodes.sample_pmatrix(rng=rng).sum(axis=1) for _ in range(50)]
    ... ).mean(0)
    >>> jnp.allclose(kbar, expected, rtol=1e-1).item()
    True

    Integral is jit-compatible and differentiable.
    >>> @jax.jit
    ... def compute_kbar(model):
    ...     return DegreeIntegral.from_model(model).integrate()[0]
    >>> jnp.allclose(compute_kbar(model), kbar, rtol=1e-5).item()
    True
    >>> grad = jax.jacobian(compute_kbar)
    >>> np.asarray(grad(model).parameters.array)
    array([[ 2.8390756e+00,  4.3376803e-01, -1.8666738e-01, ...,
            -6.6482192e-03, -7.0885383e-02, -1.5310521e-03],
           [ 4.3376803e-01,  2.1771797e+01,  4.8721835e-02, ...,
            -1.2081267e-02, -3.3174299e-02, -2.3468907e-03],
           [-1.8666738e-01,  4.8721835e-02, -3.5076842e+00, ...,
            -5.3534908e-03, -1.3598135e-01, -1.5307984e-03],
           ...,
           [-8.5827827e-02,  2.9081929e-01, -1.3207673e-01, ...,
             8.2676664e-02, -1.1980609e-02, -2.9447101e-04],
           [-9.9787414e-02,  4.1642123e-01, -2.1446073e-01, ...,
            -1.1980609e-02,  1.8283134e+00, -4.5314450e-03],
           [ 3.6122240e-02,  1.2246166e-02,  1.1097690e-02, ...,
            -2.9447101e-04, -4.5314450e-03, -1.9551054e-01]],
          shape=(100, 400), dtype=...)

    Check that it works for other dimensions and volumes.
    >>> model = (
    ...     GRGG(n, Sphere(4, r=1.0)) +
    ...     Similarity(rng.normal(n), rng.normal(n)**2) +
    ...     Complementarity(rng.normal(n), rng.normal(n)**2)
    ... )
    >>> degree = DegreeIntegral.from_model(model)
    >>> kbar, _ = degree.integrate()
    >>> expected = jnp.stack(
    ...     [model.nodes.sample_pmatrix(rng=rng).sum(axis=1) for _ in range(50)]
    ... ).mean(0)
    >>> jnp.allclose(kbar, expected, rtol=1e-1).item()
    True
    """

    batch_size: int = eqx.field(static=True)

    def __init__(
        self, *args: Any, batch_size: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = self.model._get_batch_size(batch_size)

    @property
    def constant(self) -> float:
        return super().constant / self.model.n_nodes

    def define_integrand(self) -> IntegrandT:
        inner_sum = self._define_inner_sum()

        @jax.jit
        def integrand(theta: jnp.ndarray) -> jnp.ndarray:
            """Compute the integrand at given angle(s) `x`."""
            d = self.model.manifold.dim
            g = theta * self.model.manifold.linear_size
            if self.model.n_units > self.batch_size:
                _, P = jax.lax.scan(
                    inner_sum,
                    (self.model, 0, g),
                    length=number_of_batches(self.model.n_units, self.batch_size),
                )
            else:
                P = inner_sum(g)
            integral = jnp.sin(theta) ** (d - 1) * P
            return integral.reshape(-1)[: self.model.n_units]

        return integrand

    def _define_inner_sum(
        self,
    ) -> InnerSumT | InnerSumScanT:
        """Define the inner summation over nodes."""
        if self.model.n_units <= self.batch_size:

            @partial(jax.checkpoint)
            @partial(jax.jit, donate_argnums=0)
            def inner_sum(g: jnp.ndarray) -> jnp.ndarray:
                probs = self.nodes.pairs.probs(g, adjust_quantized=True)
                if self.model.is_quantized:
                    probs *= self.model.parameters.weights
                return probs.sum(axis=-1)
        else:

            @partial(jax.checkpoint, prevent_cse=False)
            @partial(jax.jit, donate_argnums=0)
            def inner_sum(
                carry: HeterogeneousCarryT,
                *args: Any,  # noqa
            ) -> tuple[HeterogeneousCarryT, jnp.ndarray]:
                model, i, g = carry
                indices = i + jnp.arange(self.batch_size)
                p = model.pairs[indices].probs(g, adjust_quantized=True).sum(axis=-1)
                return (model, i + self.batch_size, g), p

        return inner_sum
