from collections.abc import Mapping, Sequence
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self

import jax.numpy as np

from ._lazy import LazyBroadcast, LazyOuter

if TYPE_CHECKING:
    from .abc import AbstractModelModule


class NodePairs:
    """Node pairs indexer.

    Attributes
    ----------
    module
        Parent model module.
    """

    def __init__(self, module: "AbstractModelModule") -> None:
        self.module = module
        self._args = None

    def __getitem__(self, args: Any) -> Self:
        self._args = args
        return self

    @property
    def beta(self) -> LazyOuter | list[LazyOuter]:
        """Beta parameter outer product."""
        return self._get_param(self.module.beta, "beta")

    @property
    def mu(self) -> LazyOuter | list[LazyOuter]:
        """Mu parameter outer product."""
        return self._get_param(self.module.mu, "mu")

    def probs(self, g: np.ndarray) -> np.ndarray:
        """Compute pairwise connection probabilities.

        Parameters
        ----------
        g
            Pairwise distances.
        """
        return self._get_probs(self.module.parameters, g)

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> LazyOuter:
        return params[name].outer[self._args]

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[np.ndarray]:
        return [self._get_param(p, name) for p in params]

    @singledispatchmethod
    def _get_probs(self, params: Mapping, g: np.ndarray) -> np.ndarray:
        beta = params["beta"].value
        if np.isscalar(beta):
            beta = beta[..., None]
        mu = params["mu"].value
        return LazyBroadcast(g, beta, mu, self.module._function)[self._args]

    # @_get_probs.register
    # def _(self, params: Sequence, g: np.ndarray) -> list[np.ndarray]:
