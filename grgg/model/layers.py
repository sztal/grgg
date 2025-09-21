from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Self

import equinox as eqx
import jax.numpy as jnp

from grgg import options
from grgg._typing import Floats
from grgg.manifolds import CompactManifold
from grgg.model.abc import ParamsT

from .abc import AbstractModelModule
from .functions import CouplingFunction, ProbabilityFunction
from .parameters import Beta, Mu, Parameters

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

    beta: jnp.ndarray
    mu: jnp.ndarray
    log: bool = eqx.field(static=True)
    eps: float = eqx.field(static=True, repr=False)
    function: Callable[[Floats, Floats, Floats], Floats] | None = eqx.field(
        static=True,
        repr=False,
        init=False,
    )
    _model_getter: Callable[[], "GRGG"] | None = eqx.field(static=True, repr=False)

    def __init__(
        self,
        beta: jnp.ndarray | None = None,
        mu: jnp.ndarray | None = None,
        *,
        log: bool | None = None,
        eps: float | None = None,
        _model_getter: Callable[[], "GRGG"] | None = None,
    ) -> None:
        """Initialize layer."""
        self.beta = Beta.validate(beta)
        self.mu = Mu.validate(mu)
        self.log = bool(options.model.log if log is None else log)
        self.eps = float(options.model.eps if eps is None else eps)
        self._model_getter = _model_getter
        self.function = self.define_function(self.model) if self._model_getter else None

    def __check_init__(self) -> None:
        if self.eps <= 0:
            errmsg = "'eps' must be positive"
            raise ValueError(errmsg)
        if self._model_getter is not None:
            self.parameters.validate(self.n_units)

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
        return Parameters(beta=self.beta, mu=self.mu)

    @property
    def model(self) -> "GRGG":
        """The GRGG model this layer is part of."""
        if self._model_getter is None:
            errmsg = "layer is not linked to a model"
            raise AttributeError(errmsg)
        return self._model_getter()

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the model."""
        return self.model.n_nodes

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.model.n_units

    @property
    def is_heterogeneous(self) -> bool:
        """Whether the layer has heterogeneous parameters."""
        return any(not jnp.isscalar(param) for param in self.parameters.values())

    @property
    def is_quantized(self) -> bool:
        """Whether the layer has quantized parameters."""
        return self.model.is_quantized

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
            and jnp.array_equal(self.mu, other.mu)
            and jnp.array_equal(self.beta, other.beta)
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

        def _model_getter() -> "GRGG":
            return model

        return self.replace(_model_getter=_model_getter)

    def detach(self) -> Self:
        """Return a shallow copy detached from any model."""
        return self.replace(_model_getter=None)

    def set_parameters(
        self, parameters: ParamsT | None = None, **kwargs: jnp.ndarray
    ) -> Self:
        if parameters is None:
            parameters = {}
        parameters = {**parameters, **kwargs}
        if any(key not in self.parameters for key in parameters):
            errmsg = "invalid parameter name(s) provided"
            raise KeyError(errmsg)
        return self.replace(**parameters) if parameters else self

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
