from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from .model import GRGG


class KBarOptimizer:
    r"""Optimizer for solving for average degree, :math:`\bar{k}`

    Attributes
    ----------
    kbar
        The target average degree.
    weights
        Relative kernel strength weights.
        By default, all kernels are equally weighted.
    """

    def __init__(self, model: "GRGG", kbar: float, weights: np.ndarray) -> None:
        self.model = model
        self.kbar = kbar
        self.weights = self._make_weights(weights)

    def loss(self) -> float:
        """Compute the loss function for the objective function."""
        kbar_model = self.model.kbar
        loss = (kbar_model - self.kbar) ** 2
        if self.weights.size > 1:
            K = [submodel.kbar for submodel in self.model.submodels]
            K_target = self.weights * sum(K)
            loss += np.sum((K - K_target) ** 2)
        return loss

    def objective(self, mu: np.ndarray) -> float:
        """Objective function for optimization."""
        self.model.set_mu(mu)
        return self.loss()

    def optimize(
        self, x0: np.ndarray | None = None, method: str = "Nelder-Mead", **kwargs: Any
    ) -> np.ndarray:
        """Optimize the model to achieve the target average degree.

        Parameters
        ----------
        x0
            Initial guess for the optimization.
            If None, defaults to the current mu values of the kernels.
        method
            Optimization method to use, default is 'Nelder-Mead'.
        **kwargs
            Additional keyword arguments passed to the
            :func:`scipy.optimize.minimize` function.
        """
        if x0 is None:
            x0 = np.array([kernel.mu for kernel in self.model.kernels])

        solution = minimize(self.objective, x0, method=method, **kwargs)
        if not solution.success:
            errmsg = f"optimization failed: {solution.message}"
            raise RuntimeError(errmsg)
        return solution.x

    # Internals ----------------------------------------------------------------------

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
