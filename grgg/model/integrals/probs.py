import jax.numpy as jnp

from .abc import AbstractPairsIntegral

__all__ = ("EdgeProbabilityIntegral",)


class EdgeProbabilityIntegral(AbstractPairsIntegral):
    """Edge probability integral.

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
        return super().constant / self.model.n_nodes

    def integrand(self, theta: jnp.ndarray) -> jnp.ndarray:
        d = self.model.manifold.dim
        R = self.model.manifold.linear_size
        return jnp.sin(theta) ** (d - 1) * self.pairs.probs(theta * R)


@EdgeProbabilityIntegral.register_homogeneous
class HomogeneousEdgeProbabilityIntegral(EdgeProbabilityIntegral):
    """Homogeneous edge probability integral.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg import GRGG, Similarity, Complementarity
    >>> from grgg.model.integrals.probs import EdgeProbabilityIntegral
    >>> model = GRGG(100, 2) + Similarity + Complementarity  # homogeneous
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> prob = integral.integrate()[0]
    >>> prob.item()
    0.12060938
    >>> P = jnp.array([model.sample_pmatrix(condensed=True).mean() for _ in range(100)])
    >>> jnp.isclose(prob, P.mean(), rtol=1e-2).item()
    True

    Check that this works for other non-standard volumes as well.
    >>> from grgg import Sphere
    >>> model = GRGG(100, Sphere(2, 2.0)) + Similarity + Complementarity  # homogeneous
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> prob = integral.integrate()[0]
    >>> prob.item()
    0.21692198
    >>> P = jnp.array([model.sample_pmatrix(condensed=True).mean() for _ in range(100)])
    >>> jnp.isclose(prob, P.mean(), rtol=1e-2).item()
    True
    """


@EdgeProbabilityIntegral.register_heterogeneous
class HeterogeneousEdgeProbabilityIntegral(EdgeProbabilityIntegral):
    """Heterogeneous edge probability integral.

    Attributes
    ----------
    model
        The model the integral is defined on.

    Notes
    -----
    The integration is always done on the unit sphere, and only rescaled to the manifold
    radius in the constant multiplier. Thus, the domain of integration is `[0, pi]`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg import GRGG, Similarity, Complementarity, RandomGenerator
    >>> from grgg.model.integrals.probs import EdgeProbabilityIntegral
    >>> def error(x, y):  # relative error
    ...     return (jnp.linalg.norm(x - y) / jnp.linalg.norm(x)).item()
    >>> rng = RandomGenerator(42)
    >>> n = 100
    >>> model = (
    ...     GRGG(n, 2) +
    ...     Similarity(rng.normal(n)**2, rng.normal(n)) +
    ...     Complementarity(rng.normal(n)**2, rng.normal(n))
    ... )
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> probs = integral.integrate()[0]
    >>> P = jnp.stack([model.sample_pmatrix() for _ in range(500)]).mean(0)
    >>> jnp.isclose(probs.mean(), P.mean(), rtol=1e-3).item()
    True
    >>> error(P, probs) < 0.02
    True

    Check that this works for other non-standard volumes as well.
    >>> from grgg import Sphere
    >>> n = 100
    >>> model = (
    ...     GRGG(n, Sphere(2, 2.0)) +
    ...     Similarity(rng.normal(n)**2, rng.normal(n)) +
    ...     Complementarity(rng.normal(n)**2, rng.normal(n))
    ... )
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> probs = integral.integrate()[0]
    >>> P = jnp.stack([model.sample_pmatrix() for _ in range(500)]).mean(0)
    >>> jnp.isclose(probs.mean(), P.mean(), rtol=1e-3).item()
    True
    >>> error(P, probs) < 0.02
    True
    """
