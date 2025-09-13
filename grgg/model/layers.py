import weakref
from abc import abstractmethod
from collections.abc import Callable
from copy import copy
from functools import wraps
from typing import TYPE_CHECKING, Any, Self

import jax.numpy as np

from grgg import options
from grgg._typing import Floats
from grgg.manifolds import CompactManifold

from .abc import AbstractModelModule
from .functions import CouplingFunction, ProbabilityFunction
from .parameters import AbstractModelParameter, Beta, Mu

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
    Similarity( # Beta: 1 ..., Mu: 1 ..., Total: 2 ...
      beta=Beta( # 1 (...)
          value=Array(1.5, ...)
      ),
      mu=Mu( # 1 (...)
          value=Array(0., ...)
      ),
      log=...,
      eps=...
    )
    """

    def __init__(
        self,
        beta: Beta | None = None,
        mu: Mu | None = None,
        *,
        log: bool | None = None,
        eps: float | None = None,
    ) -> None:
        """Initialization method."""
        super().__init__()
        self._model: weakref.ReferenceType["GRGG"]  # noqa
        self.beta = Beta(beta)
        self.mu = Mu(mu)
        self.log = options.layer.log if log is None else log
        self.eps = options.layer.eps if eps is None else eps

    def __copy__(self, **kwargs: Any) -> Self:
        for field in ("beta", "mu", "log", "eps"):
            if field not in kwargs:
                kwargs[field] = copy(getattr(self, field))
        return self.__class__(**kwargs)

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
    def parameters(self) -> dict[str, AbstractModelParameter]:
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

    @model.setter
    def model(self, model: "GRGG") -> None:
        self._model = weakref.ref(model)
        self._validate_param(self.beta)
        self._validate_param(self.mu)
        self._function = self._define_function()
        self.__call__ = wraps(self._function)(self.__call__.__func__).__get__(self)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the model."""
        return self.model.n_nodes

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

    @abstractmethod
    def energy(self, g: Floats) -> Floats:
        """Energy function.

        Parameters
        ----------
        g
            Geodesic distances.
        """

    def define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
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
            energy = np.maximum(self.energy(g), self.eps)
            if self.log:
                energy = np.log(energy)
            return self.probability(energy, beta, mu)

        return layer_function

    def _validate_param(self, param: AbstractModelParameter) -> None:
        if param.is_heterogeneous and param.size != self.model.n_nodes:
            cn = type(param).__name__
            errmsg = (
                f"'{cn}' size ({param.size}) does not match number of nodes "
                f"({self.model.n_nodes})"
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
