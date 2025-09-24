import equinox as eqx
import jax
import jax.numpy as jnp

from grgg import options
from grgg._typing import Floats, Scalar, Vector
from grgg.functions import AbstractFunction

VectorOrScalar = Scalar | Vector

__all__ = (
    "CouplingFunction",
    "ProbabilityFunction",
)


class CouplingFunction(AbstractFunction):
    """Abstract base class for coupling functions.

    Attributes
    ----------
    dim
        Dimension of the underlying manifold.
    modified
        Whether the modified function should be used
        to allow for smooth interpolation between
        Erdős–Rényi and hard random geometric graphs.

    Examples
    --------
    >>> from jax.test_util import check_grads
    >>> energies = jnp.linspace(-1, 3, 5)
    >>> mus = jnp.linspace(-1, 3, 5)
    >>> betas = jnp.linspace(0, 3, 5)
    >>> coupling = CouplingFunction(2, modified=False)

    >>> check_grads(coupling, (energies, mus, betas), order=1)
    >>> check_grads(coupling, (energies, mus, betas), order=2)

    >>> coupling = CouplingFunction(2, modified=True)
    >>> check_grads(coupling, (energies, mus, betas), order=1)
    >>> check_grads(coupling, (energies, mus, betas), order=2)

    Broadcasting is supported.
    >>> E = energies[:, None, None]
    >>> M = mus[:, None] + mus
    >>> B = betas[:, None] + betas
    >>> thetas = coupling(E, M, B)
    >>> thetas.shape
    (5, 5, 5)
    """

    dim: int = eqx.field(static=True, converter=int)
    modified: bool = eqx.field(
        static=True,
        kw_only=True,
        default=None,
        converter=lambda x: (bool(x if x is not None else options.model.modified)),
    )

    def __call__(
        self, energy: VectorOrScalar, mu: VectorOrScalar, beta: VectorOrScalar
    ) -> Floats:
        """Coupling function of the GRGG model.

        Parameters
        ----------
        energy
            Edge energies.
        beta
            Inverse temperature parameter(s).
        mu
            Chemical potential parameter(s).
        """
        theta = beta * self.dim * (energy - mu)
        if self.modified:
            const = jnp.where(jnp.isinf(beta), 0.0, jnp.exp(-beta) * mu * (beta + 1))
            theta += const
        return theta

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.dim == other.dim
            and self.modified == other.modified
        )


class ProbabilityFunction(AbstractFunction):
    """GRGG edge probability function.

    Attributes
    ----------
    coupling
        Coupling function.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from grgg.model.functions import ProbabilityFunction, CouplingFunction
    >>> from jax.test_util import check_grads
    >>> coupling = CouplingFunction(2)
    >>> probability = ProbabilityFunction(coupling)
    >>> energies = jnp.linspace(-1, 3, 5)
    >>> mus = jnp.linspace(-1, 3, 5)
    >>> betas = jnp.linspace(0, 3, 5)
    >>> check_grads(probability, (energies, mus, betas), order=1)
    >>> check_grads(probability, (energies, mus, betas), order=2)

    Broadcasting is supported.
    >>> E = energies[:, None, None]
    >>> M = mus[:, None] + mus
    >>> B = betas[:, None] + betas
    >>> ps = probability(E, M, B)
    >>> ps.shape
    (5, 5, 5)

    By default the modified coupling function is used,
    so the model interpolates smoothly between Erdős–Rényi and hard RGGs.
    When `beta` is zero, the probability is constant, but does not have
    to be equal to zero - it depends on `mu`, so the entire ER family is recovered.
    With `mu` set to zero, the probability is exactly 0.5.
    >>> bool(jnp.allclose(probability(0, 0, 0), 0.5))
    True

    With other values of `mu`, the probability is still constant,
    but can take any value in (0, 1), depending on `mu`.
    >>> p = probability(0, 1, 0)
    >>> bool(jnp.allclose(p, 0.26894143))
    True
    >>> bool(jnp.allclose(probability(energies, 1, 0), p))
    True

    Now, the hard RGG model is recovered in the limit `beta = jnp.inf`.
    In this case, the probability is either 0 or 1, depending on the sign of
    `energy - mu`. This shows that `mu` plays the role of the characteristic
    distance (or threshold) in the hard RGG model. At the same time, `beta` plays
    the role of the inverse temperature, controlling the sharpness of the
    transition between 0 and 1. The `energy = mu` case is handled separately,
    and the probability is set to 0.5.
    >>> p = probability(energies, 1, jnp.inf)
    >>> bool(jnp.all(p[energies < 1] == 1.0))
    True
    >>> bool(jnp.all(p[energies > 1] == 0.0))
    True
    >>> bool(jnp.allclose(p[energies == 1], 0.5))
    True
    """

    coupling: CouplingFunction

    def __call__(
        self, energy: VectorOrScalar, mu: VectorOrScalar, beta: VectorOrScalar
    ) -> Floats:
        """Edge probability function of the GRGG model.

        Parameters
        ----------
        energy
            Edge energies.
        beta
            Inverse temperature parameter(s).
        mu
            Chemical potential parameter(s).
        """
        theta = self.coupling(energy, mu, beta)
        p = jax.scipy.special.expit(-theta)
        return jnp.where(jnp.isnan(p), 0.5, p)  # Handle 0/0 case

    def equals(self, other: object) -> bool:
        return super().equals(other) and self.coupling.equals(other.coupling)
