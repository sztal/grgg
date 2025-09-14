from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Floats, Scalar, Vector
from grgg.abc import AbstractModule

VectorLike = Scalar | Vector

__all__ = (
    "CouplingFunction",
    "ProbabilityFunction",
)


class AbstractModelFunction(AbstractModule):
    """Abstract base class for model functions."""

    function: Callable[[Floats, ...], Floats] = eqx.field(
        init=False, repr=False, static=True
    )
    gradient: Callable[[Floats, ...], Floats] | None = eqx.field(
        init=False, repr=False, static=True
    )
    hessian: Callable[[Floats, ...], Floats] | None = eqx.field(
        init=False, repr=False, static=True
    )

    def __post_init__(self) -> None:
        self.function = jax.jit(self.define_function())
        if self.deriv_argnums is not None:
            self.gradient = self.define_deriv(order=1)
            self.hessian = self.define_deriv(order=2)

    def __call__(self, *args: Any, **kwargs: Any) -> Floats:
        return self.function(*args, **kwargs)

    @property
    @abstractmethod
    def deriv_argnums(self) -> tuple[int, ...]:
        """Argument numbers with respect to which the derivatives are computed."""

    @abstractmethod
    def define_function(self) -> Callable[[Floats, ...], Floats]:
        """Define the function."""

    def define_deriv(self, order: int = 1) -> Callable[[Floats, ...], Floats]:
        """Define the derivative function.

        Parameters
        ----------
        order
            Order of the derivative (1 for gradient, 2 for Hessian).
        """
        order = int(order)
        if order not in (1, 2):
            errmsg = "only first and second derivatives are supported"
            raise NotImplementedError(errmsg)
        builder = jax.grad if order == 1 else jax.hessian
        deriv = jax.jit(builder(self.function, argnums=self.deriv_argnums))

        @wraps(deriv)
        def derivative(*args: Any, **kwargs: Any) -> Floats:
            return jnp.array(deriv(*args, **kwargs))

        return jax.jit(derivative)


class CouplingFunction(AbstractModelFunction):
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
    >>> betas = jnp.linspace(0, 3, 5)
    >>> mus = jnp.linspace(-1, 3, 5)
    >>> coupling = CouplingFunction(2, modified=False)

    >>> check_grads(coupling, (energies, betas, mus), order=1)
    >>> check_grads(coupling, (energies, betas, mus), order=2)

    >>> coupling = CouplingFunction(2, modified=True)
    >>> check_grads(coupling, (energies, betas, mus), order=1)
    >>> check_grads(coupling, (energies, betas, mus), order=2)

    Broadcasting is supported.
    >>> E = energies[:, None, None]
    >>> B = betas[:, None] + betas
    >>> M = mus[:, None] + mus
    >>> thetas = coupling(E, B, M)
    >>> thetas.shape
    (5, 5, 5)
    """

    dim: int = eqx.field(static=True, converter=int)
    modified: bool = eqx.field(static=True, kw_only=True, default=True, converter=bool)

    @property
    def deriv_argnums(self) -> tuple[int, ...]:
        return (1, 2)

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.dim == other.dim
            and self.modified == other.modified
        )

    def define_function(self) -> Callable[[VectorLike, VectorLike, VectorLike], Floats]:
        """Define the coupling function."""

        def coupling(energy: VectorLike, beta: VectorLike, mu: VectorLike) -> Floats:
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
                const = jnp.where(
                    jnp.isinf(beta), 0.0, jnp.exp(-beta) * mu * (beta + 1)
                )
                theta += const
            return theta

        return coupling


class ProbabilityFunction(AbstractModelFunction):
    """GRGG edge probability function.

    Attributes
    ----------
    coupling
        Coupling function.

    Examples
    --------
    >>> from jax.test_util import check_grads
    >>> coupling = CouplingFunction(2)
    >>> probability = ProbabilityFunction(coupling)
    >>> energies = jnp.linspace(-1, 3, 5)
    >>> betas = jnp.linspace(0, 3, 5)
    >>> mus = jnp.linspace(-1, 3, 5)
    >>> check_grads(probability, (energies, betas, mus), order=1)
    >>> check_grads(probability, (energies, betas, mus), order=2)

    Broadcasting is supported.
    >>> E = energies[:, None, None]
    >>> B = betas[:, None] + betas
    >>> M = mus[:, None] + mus
    >>> ps = probability(E, B, M)
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
    >>> p = probability(0, 0, 1)
    >>> bool(jnp.allclose(p, 0.26894143))
    True
    >>> bool(jnp.allclose(probability(energies, 0, 1), p))
    True

    Now, the hard RGG model is recovered in the limit `beta = jnp.inf`.
    In this case, the probability is either 0 or 1, depending on the sign of
    `energy - mu`. This shows that `mu` plays the role of the characteristic
    distance (or threshold) in the hard RGG model. At the same time, `beta` plays
    the role of the inverse temperature, controlling the sharpness of the
    transition between 0 and 1. The `energy = mu` case is handled separately,
    and the probability is set to 0.5.
    >>> p = probability(energies, jnp.inf, 1)
    >>> bool(jnp.all(p[energies < 1] == 1.0))
    True
    >>> bool(jnp.all(p[energies > 1] == 0.0))
    True
    >>> bool(jnp.allclose(p[energies == 1], 0.5))
    True
    """

    coupling: CouplingFunction = eqx.field(repr=False)

    @property
    def deriv_argnums(self) -> tuple[int, ...]:
        return (1, 2)

    def equals(self, other: object) -> bool:
        return super().equals(other) and self.coupling.equals(other.coupling)

    def define_function(self) -> Callable[[VectorLike, VectorLike, VectorLike], Floats]:
        """Define the probability function."""

        def probability(energy: VectorLike, beta: VectorLike, mu: VectorLike) -> Floats:
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
            theta = self.coupling(energy, beta, mu)
            p = jax.scipy.special.expit(-theta)
            return jnp.where(jnp.isnan(p), 0.5, p)  # Handle 0/0 case

        return probability
