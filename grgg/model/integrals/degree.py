from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import IntVector
from grgg.abc import AbstractGRGG, AbstractModule
from grgg.integrate import AbstractIntegral

if TYPE_CHECKING:
    from grgg.model.grgg import GRGG

__all__ = ("DegreeIntegral",)


class DegreeIntegral(AbstractIntegral, AbstractModule):
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

    model: "GRGG"

    @property
    def domain(self) -> tuple[float, float]:
        return (0, jnp.pi)

    @property
    @abstractmethod
    def constant(self) -> float:
        delta = self.model.delta
        R = self.model.manifold.linear_size
        d = self.model.manifold.dim
        dV = self.model.manifold.__class__(d - 1).volume
        return delta * R**d * dV

    @property
    def defaults(self) -> dict[str, Any]:
        return super().defaults

    def equals(self, other: object) -> bool:
        return super().equals(other) and self.model.equals(other.model)

    @classmethod
    def from_model(cls, model: "GRGG") -> Self:
        """Construct an appropriate degree integral from a model."""
        if not isinstance(model, AbstractGRGG):
            errmsg = "model must be a GRGG instance"
            raise TypeError(errmsg)
        if model.is_heterogeneous:
            return HeterogeneousDegreeIntegral(model)
        return HomogeneousDegreeIntegral(model)


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
        return jnp.sin(theta) ** (d - 1) * self.model.pairs.probs(theta * R)


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
    # inner_sum: Callable[[Self, jnp.ndarray, IntVector], jnp.ndarray] = eqx.field(
    #     static=True, repr=False
    # )
    # _batch_indices: jnp.ndarray = eqx.field(repr=False)

    def __init__(
        self, *args: Any, batch_size: int | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.batch_size = self.model._get_batch_size(batch_size)

    @property
    def constant(self) -> float:
        return super().constant / self.model.n_nodes

    def integrand(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Compute the integrand at given angle(s) `x`."""
        d = self.model.manifold.dim
        R = self.model.manifold.linear_size
        P = self.model.pairs.probs(theta * R).sum(axis=1)
        return jnp.sin(theta) ** (d - 1) * P

    @staticmethod
    def _inner_sum(integrator: Self, theta: jnp.ndarray, i: IntVector) -> jnp.ndarray:
        """Compute the inner sum for a batch of angles."""
        R = integrator.model.manifold.linear_size
        indices = jnp.arange(integrator.batch_size) + i
        return integrator.model.pairs[indices, :].probs(theta * R).sum(axis=1)
