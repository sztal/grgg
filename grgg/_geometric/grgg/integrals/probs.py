import math

import jax
import jax.numpy as jnp

from grgg.utils.integrate import IntegrandT

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
        d = self.manifold.dim
        num = math.gamma((d + 1) / 2)
        den = math.sqrt(math.pi) * math.gamma(d / 2)
        return num / den

    def define_integrand(self) -> IntegrandT:
        @jax.jit
        def integrand(theta: jnp.ndarray) -> jnp.ndarray:
            R = self.manifold.linear_size
            d = self.manifold.dim
            p = self.pairs.probs(theta * R)
            return p * jnp.sin(theta) ** (d - 1)

        return integrand


@EdgeProbabilityIntegral.register_homogeneous
class HomogeneousEdgeProbabilityIntegral(EdgeProbabilityIntegral):
    """Homogeneous expected edge probability integral.

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
    >>> from grgg.models.grgg.integrals.probs import EdgeProbabilityIntegral
    >>> rng = RandomGenerator(0)
    >>> model = GRGG(100, 2) + Similarity + Complementarity  # homogeneous
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> prob = integral.integrate()[0]
    >>> prob.item()
    0.12060938
    >>> P = jnp.array(
    ...     [model.sample_pmatrix(rng=rng, condensed=True).mean() for _ in range(100)]
    ... )
    >>> jnp.isclose(prob, P.mean(), rtol=1e-2).item()
    True

    Check consistency with empirical edge density.
    >>> E = jnp.array([model.sample(rng=rng).G.density() for _ in range(50)])
    >>> jnp.isclose(prob, E.mean(), rtol=1e-2).item()
    True

    Check consistency with expected average node degree.
    >>> degrees = model.nodes.degree()
    >>> n = model.n_nodes
    >>> jnp.allclose(degrees, prob * (n - 1)).item()
    True

    Check that this works for other non-standard volumes and dimensions as well.
    >>> from grgg import Sphere
    >>> model = GRGG(100, Sphere(4, 2.0)) + Similarity + Complementarity  # homogeneous
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> prob = integral.integrate()[0]
    >>> prob.item()
    0.04122784

    >>> P = jnp.array(
    ...     [model.sample_pmatrix(rng=rng, condensed=True).mean() for _ in range(100)]
    ... )
    >>> jnp.isclose(prob, P.mean(), rtol=1e-2).item()
    True

    Check consistency with expected average node degree.
    >>> degrees = model.nodes.degree()
    >>> n = model.n_nodes
    >>> jnp.allclose(degrees, prob * (n - 1)).item()
    True

    Check consistency with empirical edge density.
    >>> E = jnp.array([model.sample(rng=rng).G.density() for _ in range(50)])
    >>> jnp.isclose(prob, E.mean(), rtol=1e-2).item()
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
    >>> from grgg.models.grgg.integrals.probs import EdgeProbabilityIntegral
    >>> def error(x, y):  # relative error
    ...     return (jnp.linalg.norm(x - y) / jnp.linalg.norm(x)).item()
    >>> rng = RandomGenerator(42)
    >>> n = 100
    >>> model = (
    ...     GRGG(n, 2) +
    ...     Similarity(rng.normal(n), rng.normal(n)**2) +
    ...     Complementarity(rng.normal(n), rng.normal(n)**2)
    ... )
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> probs = integral.integrate()[0]
    >>> P = jnp.stack([model.sample_pmatrix(rng=rng) for _ in range(100)]).mean(0)
    >>> jnp.isclose(probs.mean(), P.mean(), rtol=1e-2).item()
    True
    >>> error(P, probs) < 0.05
    True

    Check consistency with expected node degrees.
    >>> degrees = model.nodes.degree()
    >>> jnp.allclose(degrees, probs.sum(axis=1)).item()
    True

    Check consistency with empirical edge density.
    >>> E = jnp.array([model.sample(rng=rng).G.density() for _ in range(50)])
    >>> p = probs[jnp.tril_indices_from(probs, -1)].mean()
    >>> jnp.isclose(p, E.mean(), rtol=1e-2).item()
    True

    Check that this works for other dimensions and non-standard volumes as well.
    >>> from grgg import Sphere
    >>> n = 100
    >>> model = (
    ...     GRGG(n, Sphere(4, 2.0)) +
    ...     Similarity(rng.normal(n), rng.normal(n)**2) +
    ...     Complementarity(rng.normal(n), rng.normal(n)**2)
    ... )
    >>> integral = EdgeProbabilityIntegral.from_model(model)
    >>> probs = integral.integrate()[0]
    >>> P = jnp.stack([model.sample_pmatrix(rng=rng) for _ in range(100)]).mean(0)
    >>> jnp.isclose(probs.mean(), P.mean(), rtol=1e-2).item()
    True
    >>> error(P, probs) < 0.05
    True

    Check consistency with expected node degrees.
    >>> degrees = model.nodes.degree()
    >>> jnp.allclose(degrees, probs.sum(axis=1), rtol=1e-5).item()
    True

    Check consistency with empirical edge density.
    >>> E = jnp.array([model.sample(rng=rng).G.density() for _ in range(50)])
    >>> p = probs[jnp.tril_indices_from(probs, -1)].mean()
    >>> jnp.isclose(p, E.mean(), rtol=1e-2).item()
    True
    """
