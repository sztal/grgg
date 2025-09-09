from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Function(ABC):
    """Abstract base class for mathematical functions."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the function."""

    @property
    def domain(self) -> tuple[float, float]:
        """Return the domain of the function as a tuple `(a, b)`."""
        return (-np.inf, np.inf)

    def gradient(self, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa
        """Return the gradient of the function."""
        errmsg = f"Gradient not implemented for '{self.__class__.__name__}'."
        raise NotImplementedError(errmsg)

    def hessian(self, *args: Any, **kwargs: Any) -> np.ndarray:  # noqa
        """Return the Hessian of the function."""
        errmsg = f"Hessian not implemented for '{self.__class__.__name__}'."
        raise NotImplementedError(errmsg)

    def validate_arguments(self, *args: Any, **kwargs: Any) -> None:  # noqa
        """Validate the arguments passed to the equation."""
        errmsg = f"validation not implemented for '{self.__class__.__name__}'"
        raise NotImplementedError(errmsg)
