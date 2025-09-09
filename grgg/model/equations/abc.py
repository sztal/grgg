from abc import abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod, wraps
from typing import TYPE_CHECKING, Any

import numpy as np

from grgg.functions import Function
from grgg.manifolds import CompactManifold, Sphere

if TYPE_CHECKING:
    from grgg.model import GRGG


class ModelFunction(Function):
    """Abstract base class for functions used in GRGG model equations.

    Attributes
    ----------
    model
        GRGG model instance.
    """

    def __init__(self, model: "GRGG") -> None:
        self.model = model

    @property
    def manifold(self) -> "CompactManifold":
        """Return the manifold associated with the model."""
        return self.model.manifold

    @property
    def domain(self) -> tuple[float, float]:
        """Return the domain of the equation as a tuple `(a, b)`."""
        return self._domain(self.manifold)

    @singledispatchmethod
    def _domain(self, manifold: CompactManifold) -> tuple[float, float]:
        errmsg = f"Domain not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @_domain.register
    def _(self, manifold: Sphere) -> tuple[float, float]:  # noqa
        return (0.0, np.pi)


class ModelEquation(ModelFunction):
    """Abstract base class for equations.

    Attributes
    ----------
    model
        GRGG model instance.
    """

    def __init__(self, model: "GRGG") -> None:
        super().__init__(model)
        self._domain = self.define_domain(self.manifold)
        self._constant = self.define_constant(self.manifold)
        for part in ("affine", "function", "equation"):
            definition = getattr(self, f"define_{part}")
            definition = definition(self.manifold).__get__(self)
            setattr(self, f"_{part}", definition)
            method = getattr(self, part)
            setattr(self, part, wraps(definition)(method))

    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the equation."""
        return self._equation(*args, **kwargs)

    @property
    def constant(self) -> float:
        """Return the constant term of the equation."""
        return self._constant

    def affine(self, *args: Any, **kwargs: Any) -> float:
        """Return the affine transformation of the argument-dependent part."""
        return self._affine(*args, **kwargs)

    def function(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the part of the equation which is a function of arguments."""
        return self._function(*args, **kwargs)

    def equation(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the full equation."""
        return self._equation(*args, **kwargs)

    @abstractmethod
    def define_domain(
        self, manifold: CompactManifold
    ) -> Callable[[CompactManifold], tuple[float, float]]:
        """Define the domain of the equation."""
        errmsg = f"Domain not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @abstractmethod
    def define_constant(
        self, manifold: CompactManifold
    ) -> Callable[[np.ndarray, ...], np.ndarray]:
        """Define the constant term of the equation."""
        errmsg = f"Constant not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @abstractmethod
    def define_affine(
        self, manifold: CompactManifold
    ) -> Callable[[np.ndarray, ...], np.ndarray]:
        """Define the affine part of the equation."""
        errmsg = f"Affine not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @abstractmethod
    def define_function(
        self, manifold: CompactManifold
    ) -> Callable[[np.ndarray, ...], np.ndarray]:
        """Define the function part of the equation."""
        errmsg = f"Function not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)

    @abstractmethod
    def define_equation(
        self, manifold: CompactManifold
    ) -> Callable[[np.ndarray, ...], np.ndarray]:
        """Define the full equation."""
        errmsg = f"Equation not implemented for '{manifold}'."
        raise NotImplementedError(errmsg)
