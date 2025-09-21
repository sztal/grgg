from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg.utils import number_of_batches

from .abc import AbstractNodesIntegral

if TYPE_CHECKING:
    from grgg.model.grgg import GRGG

__all__ = ("DegreeIntegral",)

HeterogeneousCarryT = tuple["GRGG", int, jnp.ndarray]


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
    >>> import numpy as np
    >>> from grgg import GRGG, Similarity, Complementarity, RandomGenerator
    >>> from grgg.model.integrals import DegreeIntegral
    >>> rng = RandomGenerator(42)
    >>> model = GRGG(100, 3, Similarity, Complementarity)  # homogeneous
    >>> degree = DegreeIntegral.from_model(model)
    >>> kbar, _ = degree.integrate()
    >>> expected = jnp.asarray(
    ...     [model.nodes.sample_pmatrix(rng=rng).sum(axis=1).mean() for _ in range(20)]
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
    >>> np.asarray(grad(model).parameters.array)
    array([[-4.1539392, 15.003034 , -4.1539392, 15.003028 ]], dtype=...)
    """

    @property
    def constant(self) -> float:
        n = self.model.n_nodes
        return super().constant * (n - 1) / n

    def integrand(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the integrand at given angle(s) `theta`."""
        d = self.model.manifold.dim
        R = self.model.manifold.linear_size
        return jnp.sin(theta) ** (d - 1) * self.nodes.pairs.probs(theta * R)


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
    >>> from grgg import GRGG, Similarity, Complementarity, RandomGenerator
    >>> from grgg.model.integrals import DegreeIntegral
    >>> rng = RandomGenerator(42)
    >>> n = 100
    >>> model = (
    ...     GRGG(n, 2) +
    ...     Similarity(rng.normal(n)**2, rng.normal(n)) +
    ...     Complementarity(rng.normal(n)**2, rng.normal(n))
    ... )
    >>> degree = DegreeIntegral.from_model(model)
    >>> kbar, _ = degree.integrate()
    >>> expected = jnp.stack(
    ...     [model.nodes.sample_pmatrix(rng=rng).sum(axis=1) for _ in range(20)]
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
    array([[-1.6744738e+01, -6.1276048e-02, -7.5484417e-02, ...,
            -4.8600473e-02,  6.8682708e-02,  2.8153904e-02],
           [-6.1276048e-02, -1.0947376e+01, -1.4008153e-02, ...,
            -6.6341758e-02,  6.1861079e-02,  3.3777410e-03],
           [-7.5484417e-02, -1.4008153e-02, -2.3531334e+00, ...,
             1.9964302e-01,  1.7936581e-01,  3.5689750e-03],
           ...,
           [-2.9722819e-01, -3.1819928e-01, -1.0324293e-01, ...,
             4.6899362e+00,  4.7270000e-02,  7.4549869e-02],
           [-1.5679037e-02, -2.2483882e-03, -9.7606033e-03, ...,
             4.7270000e-02,  1.4273865e+01,  3.4715172e-02],
           [-5.2480883e-04, -5.5024851e-05, -4.2580199e-04, ...,
             7.4549869e-02,  3.4715172e-02,  3.3579695e+00]],
          shape=(100, 400), dtype=...)
    """

    batch_size: int = eqx.field(static=True)
    inner_sum: Callable[
        [HeterogeneousCarryT, ...], tuple[HeterogeneousCarryT, jnp.ndarray]
    ] = eqx.field(static=True, init=False, repr=False)

    def __init__(
        self, *args: Any, batch_size: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = self.model._get_batch_size(batch_size)
        self.inner_sum = self._define_inner_sum()

    @property
    def constant(self) -> float:
        return super().constant / self.model.n_nodes

    def integrand(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the integrand at given angle(s) `x`."""
        d = self.model.manifold.dim
        g = theta * self.model.manifold.linear_size
        if self.model.n_units > self.batch_size:
            _, P = jax.lax.scan(
                self.inner_sum,
                (self.model, 0, g),
                length=number_of_batches(self.model.n_units, self.batch_size),
            )
        else:
            P = self.inner_sum(g)
        integral = jnp.sin(theta) ** (d - 1) * P
        return integral.reshape(-1)[: self.model.n_units]

    def _define_inner_sum(
        self,
    ) -> Callable[[HeterogeneousCarryT, ...], tuple[HeterogeneousCarryT, jnp.ndarray]]:
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
