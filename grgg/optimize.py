from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .model import GRGG


class GRGGOptimization:
    """Namespace for optimization methods in GRGG model.

    Attributes
    ----------
    kbar
        Optimizer for computing the average degree :math:`\bar{k}`.
    """

    def __init__(self, model: "GRGG") -> None:
        self._model = model

    def kbar(self, kbar: float, *args: Any, **kwargs: Any) -> np.ndarray:
        weights = kwargs.pop("weights", None)
        optimizer = KBarOptimizer(self._model.copy(), kbar, weights=weights)
        return optimizer.optimize(*args, **kwargs)


class GRGGOptimizer(ABC):
    """Abstract base class for optimization in GRGG model."""

    def __init__(self, model: "GRGG", *, weights: np.ndarray | None = None) -> None:
        self.model = model
        self.weights = self._make_weights(weights)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Call the optimizer to minimize the objective function."""
        return self.optimize(*args, **kwargs)

    @abstractmethod
    def loss(self) -> float:
        """Compute the loss function for the objective function."""

    @abstractmethod
    def get_x0(self) -> np.ndarray:
        """Get the initial guess for the optimization parameters."""

    def optimize(
        self,
        x0: np.ndarray | None = None,
        method: str = "Nelder-Mead",
        **kwargs: Any,
    ) -> np.ndarray:
        """Optimize the model parameters to minimize the loss function.

        Parameters
        ----------
        kbar
            Target average degree.
        x0
            Initial guess for the optimization.
            If None, defaults to :meth:`get_x0`.
        method
            Optimization method to use.
        **kwargs
            Additional keyword arguments passed to the
            :func:`scipy.optimize.minimize` function.
        """
        if x0 is None:
            x0 = self.get_x0()
        solution = minimize(self.objective, x0, method=method, **kwargs)
        if not solution.success:
            errmsg = f"optimization failed: {solution.message}"
            raise RuntimeError(errmsg)
        return solution.x

    def _make_weights(self, weights: np.ndarray | None = None) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(self.model.kernels))
        weights = np.atleast_1d(weights)
        if weights.ndim != 1:
            errmsg = "'weights' must be a 1D array-like"
            raise ValueError(errmsg)
        if weights.size != len(self.model.kernels):
            errmsg = "'weights' must have the same length as 'kernels'"
            raise ValueError(errmsg)
        weights = weights / weights.sum()
        return weights


class KBarOptimizer(GRGGOptimizer):
    r"""Optimizer for solving for average degree, :math:`\bar{k}`

    Attributes
    ----------
    kbar
        The target average degree.
    weights
        Relative kernel strength weights.
        By default, all kernels are equally weighted.
    """

    def __init__(self, model: "GRGG", kbar: float, **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.kbar = kbar

    def loss(self) -> float:
        """Compute the loss function for the objective function."""
        kbar_model = self.model.kbar
        loss = (kbar_model - self.kbar) ** 2
        if self.weights.size > 1:
            K = [submodel.kbar for submodel in self.model.submodels]
            K_target = self.weights * sum(K)
            loss += np.sum((K - K_target) ** 2)
        return loss

    def get_x0(self) -> np.ndarray:
        """Get the initial guess for the optimization parameters."""
        return np.array([kernel.mu for kernel in self.model.kernels])

    def objective(self, mu: np.ndarray) -> float:
        """Objective function for optimization."""
        self.model.set_kernel_params(mu=mu)
        return self.loss()
