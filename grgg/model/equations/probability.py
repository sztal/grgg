import numpy as np

from .abc import ModelFunction


class EdgeCoupling(ModelFunction):
    """Edge coupling function."""

    def __call__(
        self, energy: np.ndarray, beta: np.ndarray, mu: np.ndarray
    ) -> np.ndarray:
        """Evaluate the edge coupling function.

        Parameters
        ----------
        energy
            Edge energy.
        beta
            Inverse temperature parameter :math:`\\beta`.
            It controls the coupling between the network topology and the geometry.
        mu
            Coupling parameter :math:`\\mu`.
            It controls the network density.
        """
        d = self.manifold.dim
        coupling = beta * d * (energy - mu) + np.exp(-beta) * mu * (beta + 1)
        return coupling


class EdgeProbability(ModelFunction):
    """Abstract base class for edge probability functions."""

    def __call__(self, g: np.ndarray, beta: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Evaluate the edge probability function.

        Parameters
        ----------
        g
            Geodesic distances between nodes.
        beta
            Inverse temperature parameter :math:`\\beta`.
            It controls the coupling between the network topology and the geometry.
        mu
            Coupling parameter :math:`\\mu`.
            It controls the network density.
        """
