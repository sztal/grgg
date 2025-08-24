from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

from . import options

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
        self.__model = model

    @property
    def mu(self) -> "MuOptimizer":
        return MuOptimizer(self.__model)


class GRGGOptimizer(ABC):
    """Abstract base class for optimization in GRGG model."""

    def __init__(self, model: "GRGG", *, tol: float | None = None) -> None:
        self.model = model.copy()
        self.tol = tol if tol is not None else options.optimize.tol

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Call the optimizer to minimize the objective function."""
        return self.optimize(*args, **kwargs)

    @abstractmethod
    def loss(self, target: float | np.ndarray, weights: np.ndarray) -> float:
        """Compute the loss function for the objective function."""

    @abstractmethod
    def make_objective(
        self,
        target: float | np.ndarray,
        weights: np.ndarray,
    ) -> Callable[[np.ndarray], float]:
        """Objective function for optimization."""

    @abstractmethod
    def get_x0(self) -> np.ndarray:
        """Get the initial guess for the optimization parameters."""

    def optimize(
        self,
        target: float | np.ndarray,
        weights: np.ndarray | None = None,
        *,
        x0: np.ndarray | None = None,
        method: str | None = None,
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
        method = method or options.optimize.method
        weights = self.make_weights(weights)
        objective = self.make_objective(target, weights)
        if x0 is None:
            x0 = self.get_x0()
        solution = minimize(objective, x0, method=method, **kwargs)
        if not solution.success:
            errmsg = f"optimization failed: {solution.message}"
            raise RuntimeError(errmsg)
        if solution.fun > self.tol:
            errmsg = f"optimization loss to high: {solution.fun} > {self.tol}"
            raise RuntimeError(errmsg)
        return solution

    def make_weights(self, weights: np.ndarray | None = None) -> np.ndarray:
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


class MuOptimizer(GRGGOptimizer):
    r"""Optimizer for solving for :math:`\mu` given average degree, :math:`\bar{k}`."""

    def __init__(
        self, *args: Any, beta_max: float | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        # Work with finite beta values during optimization
        beta_max = beta_max if beta_max is not None else options.optimize.mu.beta_max
        for kernel in self.model.kernels:
            kernel.beta = min(kernel.beta, beta_max)

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        try:
            return self.optimize(*args, **kwargs)
        except RuntimeError as exc:
            if np.allclose(kwargs.get("x0", []), 0):
                kwargs["x0"] = np.zeros(len(self.model.kernels))
                return self.optimize(*args, **kwargs)
            raise exc

    def loss(self, target: float, weights: np.ndarray) -> float:
        """Compute the loss function for the objective function."""
        kbar_model = self.model.kbar
        loss = (kbar_model - target) ** 2
        if weights.size > 1:
            K = [submodel.kbar for submodel in self.model.submodels]
            K_target = weights * sum(K)
            loss += np.sum((K - K_target) ** 2)
        return loss

    def make_objective(
        self,
        target: float | np.ndarray,
        weights: np.ndarray,
    ) -> Callable[[np.ndarray], float]:
        """Objective function for optimization."""

        def objective(mu: np.ndarray) -> float:
            """Objective function to minimize."""
            self.model.set_kernel_params(mu=mu)
            return self.loss(target, weights)

        return objective

    def get_x0(self) -> np.ndarray:
        """Get the initial guess for the optimization parameters."""
        return np.array([kernel.mu for kernel in self.model.kernels])
