from abc import abstractmethod
from collections.abc import Mapping, Sequence
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self

import jax.numpy as np
from flax import nnx

from grgg.lazy import LazyOuter

if TYPE_CHECKING:
    from .abc import AbstractModelModule


class AbstractNodeIndexer(nnx.Module):
    """Abstract base class for node indexers."""

    def __init__(self, module: "AbstractModelModule") -> None:
        super().__init__()
        self.module = module

    @abstractmethod
    def __getitem__(self, args: Any) -> Self:
        """Indexing method."""

    @property
    @abstractmethod
    def beta(self) -> np.ndarray | LazyOuter | list[LazyOuter]:
        """Beta parameter outer product."""

    @property
    @abstractmethod
    def mu(self) -> np.ndarray | LazyOuter | list[LazyOuter]:
        """Mu parameter outer product."""


class NodeIndexer(AbstractNodeIndexer):
    """Node indexer.

    Helper class for indexing model parameters and computing node-specific
    quantities for specific node selections.

    Attributes
    ----------
    module
        Parent model module.
    """

    def __init__(self, module: "AbstractModelModule") -> None:
        super().__init__(module)
        self._i = None

    def __getitem__(self, args: Any) -> Self:
        if self._i is None:
            self._i = args
        else:
            errmsg = "too many indices for nodes"
            raise IndexError(errmsg)
        return self

    @property
    def beta(self) -> np.ndarray:
        """Beta parameter outer product."""
        return self._get_param(self.module.parameters, "beta")

    @property
    def mu(self) -> np.ndarray:
        """Mu parameter outer product."""
        return self._get_param(self.module.parameters, "mu")

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> np.ndarray:
        """Get parameter outer product."""
        param = params[name]
        if self._i is not None:
            return param[self._i]
        return param

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[np.ndarray]:
        return [self._get_param(p, name) for p in params]


class NodePairIndexer(AbstractNodeIndexer):
    """Node pairs indexer.

    Helper class for indexing model parameters and computing pairwise
    connection probabilities and other quantities for specific node pair
    selections.

    Attributes
    ----------
    module
        Parent model module.

    Examples
    --------
    Indexing in the homogeneous case always returns scalars.
    >>> from grgg import GRGG, Similarity, Complementarity
    >>> model = GRGG(100, 2, Similarity(2, 1), Complementarity(1, 0))
    >>> model.pairs[0, 1].beta
    [Array(2., ...), Array(1., ...]
    >>> model.pairs[[0, 1], [1, 0]].mu
    [Array(1., ...), Array(0., ...)]

    A specific layer can be indexed too.
    >>> model.layers[0].pairs[0, 1].beta
    Array(2., ...)

    In the heterogeneous case, indexing may return larger arrays.
    >>> model = GRGG(3, 2, Similarity([1,2,3], [4,5,6]))
    >>> model.pairs[0, 1].beta
    [Array(3., ...)]
    >>> model.pairs[...].mu
    [Array([[ 8.,  9., 10.],
            [ 9., 10., 11.],
            [10., 11., 12.]], ...)]

    Indexing with cartesian indices is supported too
    as supported by :mod:`numpy` and :mod:`jax`.
    >>> model.layers[0].pairs[[0, 1], [1, 2]].mu
    Array([ 9., 11.], ...)

    Selecting rectangular blocks is also supported through repeated indexing.
    >>> model.layers[0].pairs[[0, 1]][[1, 2]].mu
    Array([[ 9., 10.],
           [10., 11.]], ...)
    """

    def __init__(self, module: "AbstractModelModule") -> None:
        self.module = module
        self._i = None
        self._j = None

    def __getitem__(self, args: Any) -> Self:
        if self._i is None:
            self._i = args
        elif self._j is None:
            self._j = args
        else:
            errmsg = "too many indices for node pairs"
            raise IndexError(errmsg)
        return self

    @property
    def beta(self) -> LazyOuter | list[LazyOuter]:
        """Beta parameter outer product."""
        return self._get_param(self.module.parameters, "beta")

    @property
    def mu(self) -> LazyOuter | list[LazyOuter]:
        """Mu parameter outer product."""
        return self._get_param(self.module.parameters, "mu")

    def probs(self, g: np.ndarray) -> np.ndarray:
        """Compute pairwise connection probabilities.

        Parameters
        ----------
        g
            Pairwise distances.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = GRGG(3, 2, Similarity([1,2,3], [1,0,0]))
        >>> model.layers[0].pairs[[0, 1]][[0, 2]].probs(1)
        Array([[0.999245 , 0.9996325],
               [0.9969842, 0.5      ]], ...)

        Evaluate the mult-layer model probabilities.
        >>> model.pairs[[0, 1]][[0, 2]].probs(1)
        Array([[0.999245 , 0.9996325],
               [0.9969842, 0.5      ]], ...)
        """
        return self._get_probs(self.module.parameters, g)

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> LazyOuter:
        outer = params[name].outer[self._i]
        if self._j is not None:
            return outer[:, self._j]
        return outer

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[np.ndarray]:
        return [self._get_param(p, name) for p in params]

    @singledispatchmethod
    def _get_probs(self, params: Mapping, g: np.ndarray) -> np.ndarray:
        beta = self._get_param(params, "beta")
        mu = self._get_param(params, "mu")
        return self.module._function(g, beta, mu)

    @_get_probs.register
    def _(self, params: Sequence, g: np.ndarray) -> list[np.ndarray]:
        beta = np.stack([self._get_param(p, "beta") for p in params])
        mu = np.stack([self._get_param(p, "mu") for p in params])
        return self.module._function(g, beta, mu)
