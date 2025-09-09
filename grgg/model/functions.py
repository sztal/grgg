from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Self

import jax
import jax.numpy as np
from flax import nnx

from grgg import options
from grgg.manifolds import CompactManifold

from ._typing import Floats, Scalar, Vector
from .abc import AbstractModelElement

VectorLike = Scalar | Vector

__all__ = (
    "CouplingFunction",
    "ProbabilityFunction",
    "SimilarityFunction",
    "ComplementarityFunction",
    "LayerFunction",
)


class AbstractModelFunction(AbstractModelElement):
    """Abstract base class for model functions."""

    def __init__(self) -> None:
        super().__init__()
        self.function = nnx.jit(self.define_function())
        self.__call__ = wraps(self.function)(self.__call__.__func__).__get__(self)
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
        builder = nnx.grad if order == 1 else jax.hessian
        deriv = nnx.jit(builder(self.function, argnums=self.deriv_argnums))

        @wraps(deriv)
        def derivative(*args: Any, **kwargs: Any) -> Floats:
            return np.array(deriv(*args, **kwargs))

        return nnx.jit(derivative)


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
    >>> energies = np.linspace(-1, 3, 5)
    >>> betas = np.linspace(0, 3, 5)
    >>> mus = np.linspace(-1, 3, 5)
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

    def __init__(self, dim: int, *, modified: bool = True) -> None:
        self.dim = dim
        self.modified = modified
        super().__init__()

    def __copy__(self) -> Self:
        return type(self)(self.dim, modified=self.modified)

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
                const = np.where(np.isinf(beta), 0.0, np.exp(-beta) * mu * (beta + 1))
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
    >>> energies = np.linspace(-1, 3, 5)
    >>> betas = np.linspace(0, 3, 5)
    >>> mus = np.linspace(-1, 3, 5)
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
    to be equal to zero---it depends on `mu`, so the entire ER family is recovered.
    With `mu` set to zero, the probability is exactly 0.5.
    >>> bool(np.allclose(probability(0, 0, 0), 0.5))
    True

    With other values of `mu`, the probability is still constant,
    but can take any value in (0, 1), depending on `mu`.
    >>> p = probability(0, 0, 1)
    >>> bool(np.allclose(p, 0.26894143))
    True
    >>> bool(np.allclose(probability(energies, 0, 1), p))
    True

    Now, the hard RGG model is recovered in the limit `beta = np.inf`.
    In this case, the probability is either 0 or 1, depending on the sign of
    `energy - mu`. This shows that `mu` plays the role of the characteristic
    distance (or threshold) in the hard RGG model. At the same time, `beta` plays
    the role of the inverse temperature, controlling the sharpness of the
    transition between 0 and 1. The `energy = mu` case is handled separately,
    and the probability is set to 0.5.
    >>> p = probability(energies, np.inf, 1)
    >>> bool(np.all(p[energies < 1] == 1.0))
    True
    >>> bool(np.all(p[energies > 1] == 0.0))
    True
    >>> bool(np.allclose(p[energies == 1], 0.5))
    True
    """

    def __init__(self, coupling: CouplingFunction) -> None:
        self.coupling = coupling
        super().__init__()

    def __copy__(self) -> Self:
        return type(self)(self.coupling.__copy__())

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
            p = nnx.sigmoid(-theta)
            return np.where(np.isnan(p), 0.5, p)  # Handle 0/0 case

        return probability


class AbstractEnergyFunction(AbstractModelFunction):
    """Abstract base class for energy functions."""

    def __init__(
        self,
        *args: Any,  # noqa
        eps: float | None = None,
        log: bool | None = None,
    ) -> None:
        self.eps = float(eps if eps is not None else options.layer.eps)
        self.log = bool(log if log is not None else options.layer.log)
        super().__init__()

    def __call__(self, g: Floats) -> Floats:
        """Compute edge energies from geodesic distances."""
        energy = np.maximum(self.energy(g), self.eps)
        if self.log:
            energy = np.log(energy)
        return energy

    def __copy__(self) -> Self:
        return type(self)(self.manifold.copy(), eps=self.eps, log=self.log)

    @property
    def deriv_argnums(self) -> None:
        return None

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.manifold == other.manifold
            and self.eps == other.eps
            and self.log == other.log
        )

    def define_function(self) -> Callable[[Floats], Floats]:
        @wraps(self.energy)
        def energy(g: Floats) -> Floats:
            E = np.maximum(self.energy(g), self.eps)
            if self.log:
                E = np.log(E)
            return E

        return energy

    @abstractmethod
    def energy(self, g: Floats) -> Floats:
        """Compute edge energies from geodesic distances."""


class SimilarityFunction(AbstractEnergyFunction):
    """Similarity-based energy function."""

    def energy(self, g: Floats) -> Floats:
        return g


class ComplementarityFunction(AbstractEnergyFunction):
    """Complementarity-based energy function.

    Attributes
    ----------
    diameter
        Diameter of the underlying manifold.
        Can be passed also as a full-fledged manifold instance.
    """

    def __init__(self, diameter: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(diameter, CompactManifold):
            diameter = diameter.diameter
        self.diameter = diameter

    def energy(self, g: Floats) -> Floats:
        return self.diameter - g


class LayerFunction(AbstractModelFunction):
    """Layer probability function.

    Attributes
    ----------
    probability
        Probability function.
    energy
        Energy function.

    Examples
    --------
    Layer with similarity energy.
    >>> from jax.test_util import check_grads
    >>> geodesics = np.linspace(0, 3, 5)
    >>> betas = np.linspace(0, 3, 5)
    >>> mus = np.linspace(-1, 3, 5)
    >>> coupling = CouplingFunction(2)
    >>> energy = SimilarityFunction(log=False)
    >>> probability = ProbabilityFunction(coupling)
    >>> layer = LayerFunction(energy, probability)
    >>> check_grads(layer, (geodesics, betas, mus), order=1)

    Layer with complementarity log-energy.
    >>> energy = ComplementarityFunction(diameter=np.pi, log=True)
    >>> layer = LayerFunction(energy, probability)
    >>> check_grads(layer, (geodesics, betas, mus), order=1)
    """

    def __init__(
        self,
        energy: AbstractEnergyFunction,
        probability: ProbabilityFunction,
    ) -> None:
        self.energy = energy
        self.probability = probability
        super().__init__()

    def __copy__(self) -> Self:
        return type(self)(self.probability.__copy__(), self.energy.__copy__())

    @property
    def deriv_argnums(self) -> tuple[int, ...]:
        return (1, 2)

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and self.probability.equals(other.probability)
            and self.energy.equals(other.energy)
        )

    def define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
        """Define the layer function."""

        def layer(g: Floats, beta: Floats, mu: Floats) -> Floats:
            """Compute edge probabilities from geodesic distances.

            Parameters
            ----------
            g
                Geodesic distances.
            beta
                Inverse temperature parameter(s).
            mu
                Chemical potential parameter(s.).
            """
            energy = self.energy(g)
            return self.probability(energy, beta, mu)

        return layer
