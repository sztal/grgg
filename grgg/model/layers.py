import weakref
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax.numpy as jnp

from grgg import options
from grgg._typing import Floats
from grgg.abc import AbstractGRGG
from grgg.manifolds import CompactManifold

from .abc import AbstractModelModule
from .functions import CouplingFunction, ProbabilityFunction
from .parameters import Beta, Mu

if TYPE_CHECKING:
    from .grgg import GRGG


class AbstractLayer(AbstractModelModule):
    """Abstract base class for layers.

    Attributes
    ----------
    beta
        Inverse temperature parameter(s).
    mu
        Chemical potential parameter(s).
    log
        Whether log-energy should be used.
    eps
        Small constant added to energies for numerical stability.

    Examples
    --------
    >>> Similarity()
    Similarity(beta=f...[], mu=f...[], log=...)
    """

    beta: jnp.ndarray = eqx.field(default=None, converter=Beta.validate)
    mu: jnp.ndarray = eqx.field(default=None, converter=Mu.validate)
    log: bool = eqx.field(
        default=None,
        kw_only=True,
        static=True,
        converter=lambda x: bool(options.layer.log if x is None else x),
    )
    eps: float = eqx.field(
        default=None,
        kw_only=True,
        static=True,
        repr=False,
        converter=lambda x: float(options.layer.eps if x is None else x),
    )
    _model: weakref.ReferenceType["GRGG"] | None = eqx.field(
        default=None, repr=False, static=True
    )
    function: Callable[[Floats, Floats, Floats], Floats] = eqx.field(
        default=None, repr=False, static=True
    )

    def __call__(self, g: Floats, beta: Floats, mu: Floats) -> Floats:
        """Compute layer edge probabilities.

        Parameters
        ----------
        g
            Geodesic distances.
        beta
            Inverse temperature parameter(s).
        mu
            Chemical potential parameter(s).
        """
        return self._function(g, beta, mu)

    @property
    def parameters(self) -> dict[str, jnp.ndarray]:
        """Layer parameters."""
        return {"beta": self.beta, "mu": self.mu}

    @property
    def model(self) -> "GRGG":
        """The GRGG model this layer is part of."""
        model = getattr(self, "_model", None)
        model = model() if model is not None else None
        if model is None:
            errmsg = "layer is not linked to a model"
            raise AttributeError(errmsg)
        return model

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the model."""
        return self.model.n_nodes

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the layer has heterogeneous parameters."""
        return any(not jnp.isscalar(param) for param in self.parameters.values())

    @property
    def manifold(self) -> CompactManifold:
        """The manifold the model is embedded in."""
        return self.model.manifold

    @property
    def probability(self) -> ProbabilityFunction:
        """The probability function."""
        return self.model.probability

    @property
    def coupling(self) -> CouplingFunction:
        """The coupling function."""
        return self.probability.coupling

    def equals(self, other: object) -> bool:
        """Check if two layers are equal."""
        return (
            super().equals(other)
            and self.mu.equals(other.mu)
            and self.beta.equals(other.beta)
            and self.probability.equals(other.probability)
            and self.log == other.log
            and self.eps == other.eps
        )

    def attach(self, model: "GRGG") -> Self:
        """Return a shallow copy attached to a model.

        Parameters
        ----------
        model
            The GRGG model to attach to.
        """
        if not isinstance(model, AbstractGRGG):
            errmsg = "layer can only be attached to a GRGG model"
            raise TypeError(errmsg)
        model = weakref.ref(model)  # type: ignore
        function = self.define_function(model())
        layer = self.replace(_model=model, function=function)
        layer._validate_param(layer.beta)
        layer._validate_param(layer.mu)
        return layer

    @abstractmethod
    def energy(self, g: Floats) -> Floats:
        """Energy function.

        Parameters
        ----------
        g
            Geodesic distances.
        """

    def define_function(
        self, model: "GRGG"
    ) -> Callable[[Floats, Floats, Floats], Floats]:
        """Define the layer function."""

        def layer_function(g: Floats, beta: Floats, mu: Floats) -> Floats:
            """Compute layer edge probabilities.

            Parameters
            ----------
            g
                Geodesic distances.
            beta
                Inverse temperature parameter(s).
            mu
                Chemical potential parameter(s).
            """
            energy = jnp.maximum(self.energy(g), self.eps)
            if self.log:
                energy = jnp.log(energy)
            return model.probability(energy, beta, mu)

        return layer_function

    def _validate_param(self, param: jnp.ndarray) -> None:
        if not jnp.isscalar(param) and param.size != self.model.n_nodes:
            errmsg = (
                f"node parameter size ({param.size}) does not match number of nodes "
                f"({self.n_nodes})"
            )
            raise ValueError(errmsg)


class Similarity(AbstractLayer):
    """GRGG layer with similarity-based connection probability.

    Attributes
    ----------
    beta
        Inverse temperature parameter(s).
    mu
        Chemical potential parameter(s).
    """

    def energy(self, g: Floats) -> Floats:
        r"""Similarity-based energy.

        .. math::

            \varepsilon_{ij} = g_{ij}

        Parameters
        ----------
        g
            Geodesic distances.
        """
        return g


class Complementarity(AbstractLayer):
    """GRGG layer with complementarity-based connection probability.

    Attributes
    ----------
    beta
        Inverse temperature parameter(s).
    mu
        Chemical potential parameter(s).
    """

    def energy(self, g: Floats) -> Floats:
        r"""Complementarity-based energy.

        .. math::

            \varepsilon_{ij} = g_{\max} - g_{ij}

        Parameters
        ----------
        g
            Geodesic distances.
        """
        return self.manifold.diameter - g
