import weakref
from abc import abstractmethod
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self

from grgg.utils import split_kwargs_by_signatures

from .abc import AbstractModelModule
from .functions import (
    AbstractEnergyFunction,
    ComplementarityFunction,
    CouplingFunction,
    LayerFunction,
    ProbabilityFunction,
    SimilarityFunction,
)
from .parameters import AbstractModelParameter, Beta, Mu

if TYPE_CHECKING:
    from .model import GRGG


class AbstractLayer(AbstractModelModule):
    """Abstract base class for layers.

    Attributes
    ----------
    beta
        Inverse temperature parameter(s).
    mu
        Chemical potential parameter(s).
    """

    @singledispatchmethod
    def __init__(
        self,
        beta: Beta | None = None,
        mu: Mu | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialization method.

        `**kwargs` are stored and passed to the energy and coupling
        functions when the layer is linked to a model.
        """
        self._model: weakref.ReferenceType["GRGG"]  # noqa
        self._function_kwargs = kwargs
        self.energy: AbstractEnergyFunction
        self.probability: ProbabilityFunction
        super().__init__()
        self.beta = Beta(beta)
        self.mu = Mu(mu)

    @__init__.register
    def _(self, model: "GRGG", *args: Any, **kwargs: Any) -> None:
        self.__init__(*args, **kwargs)
        self.model = model

    def __copy__(self) -> Self:
        return type(self)(self.model, self.beta.copy(), self.mu.copy())

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
        if not isinstance(model, GRGG):
            errmsg = "model must be a GRGG instance"
            raise TypeError(errmsg)
        self._model = weakref.ref(model)
        self._validate_param(self.beta)
        self._validate_param(self.mu)
        kwargs = self.__dict__.pop("_function_kwargs", {})
        ekw, ckw, *_ = split_kwargs_by_signatures(
            kwargs, AbstractEnergyFunction, CouplingFunction
        )
        energy = self.define_energy(self.model.manifold, **ekw)
        coupling = CouplingFunction(self.model.manifold.dim, **ckw)
        probability = ProbabilityFunction(coupling)
        self.probability = LayerFunction(energy, probability)

    @property
    def coupling(self) -> CouplingFunction:
        """The coupling function."""
        return self.probability.probability.coupling

    @abstractmethod
    def define_energy(self) -> AbstractEnergyFunction:
        """Define the energy function for the layer when linked to a model."""

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

    def define_energy(self) -> AbstractEnergyFunction:
        return SimilarityFunction(self.model.manifold)


class Complementarity(AbstractLayer):
    """GRGG layer with complementarity-based connection probability.

    Attributes
    ----------
    beta
        Inverse temperature parameter(s).
    mu
        Chemical potential parameter(s).
    """

    def define_energy(self) -> AbstractEnergyFunction:
        return ComplementarityFunction(self.model.manifold)
