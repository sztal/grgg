from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg.abc import AbstractGRGG
from grgg.lazy import LazyOuter
from grgg.utils import squareform

from ._sampling import Sample, Sampler

if TYPE_CHECKING:
    from .abc import AbstractModelModule
    from .grgg import GRGG


class AbstractModelView(eqx.Module):
    """Abstract base class for node indexers."""

    module: "AbstractModelModule"

    @abstractmethod
    def __getitem__(self, args: Any) -> Self:
        """Indexing method."""

    @property
    @abstractmethod
    def beta(self) -> jnp.ndarray | LazyOuter | list[LazyOuter]:
        """Beta parameter outer product."""

    @property
    @abstractmethod
    def mu(self) -> jnp.ndarray | LazyOuter | list[LazyOuter]:
        """Mu parameter outer product."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether the view is active (i.e., has any indices selected)."""

    @property
    def reindex(self) -> Self:
        """Return cleared view to allow reindexing."""
        self.clear()
        return self

    @abstractmethod
    def reset(self) -> None:
        """Reset the view to include all nodes."""


class NodeView(AbstractModelView):
    """Node view.

    Helper class for indexing model parameters and computing node-specific
    quantities for specific node selections.

    Attributes
    ----------
    module
        Parent model module.
    """

    _i: int | slice | Sequence[int] | None = eqx.field(
        default=None, repr=False, kw_only=True
    )

    def __getitem__(self, args: Any) -> Self:
        if self._i is None:
            return replace(self, _i=args)
        errmsg = "too many indices for nodes"
        raise IndexError(errmsg)

    @property
    def n_nodes(self) -> int:
        """Number of selected nodes."""
        if self._i is None:
            return self.module.n_nodes
        if isinstance(self._i, int):
            return 1
        if isinstance(self._i, slice):
            return len(range(*self._i.indices(self.module.n_nodes)))
        return len(self._i)

    @property
    def pairs(self) -> "NodePairView":
        """Node pairs view."""
        pairs = NodePairView(self.module)
        if self._i is not None:
            return pairs[self._i][self._i]
        return pairs

    @property
    def beta(self) -> jnp.ndarray:
        """Beta parameter outer product."""
        return self._get_param(self.module.parameters, "beta")

    @property
    def mu(self) -> jnp.ndarray:
        """Mu parameter outer product."""
        return self._get_param(self.module.parameters, "mu")

    @property
    def is_active(self) -> bool:
        """Whether the view is active (i.e., has any indices selected)."""
        return self._i is not None

    def reset(self) -> None:
        """Reset the current node selection."""
        self._i = None

    def probs(self, g: jnp.ndarray) -> jnp.ndarray:
        """Compute edge probabilities within the selected group of nodes."""
        return self.pairs.probs(g)

    def sample_points(self, **kwargs: Any) -> jnp.ndarray:
        """Sample points from the selected group of nodes.

        `**kwargs`* are passed to :meth:`~grgg.manifolds.Manifold.sample_points`.
        """
        return self.module.manifold.sample_points(self.n_nodes, **kwargs)

    def sample_pmatrix(self, *, condensed: bool = False, **kwargs: Any) -> jnp.ndarray:
        """Sample probability matrix from the selected group of nodes.

        `**kwargs`* are passed to :meth:`sample_points`.

        Examples
        --------
        >>> from grgg import GRGG, Similarity
        >>> model = GRGG(5, 2, Similarity(2, 1))
        >>> P = model.nodes.sample_pmatrix()  # Full probability matrix
        >>> P.shape
        (5, 5)
        >>> P_sub = model.nodes[[0, 2, 4]].sample_pmatrix()  # For nodes 0, 2, and 4
        >>> P_sub.shape
        (3, 3)
        """
        points = self.sample_points(**kwargs)
        g = self.module.manifold.distances(points, condensed=True)
        i, j = jnp.triu_indices(len(points), k=1)
        p = self.pairs[i, j].probs(g)
        return p if condensed else squareform(p)

    def sample(self, **kwargs: Any) -> Sample:
        """Generate a model sample for the selected group of nodes.

        `**kwargs`* are passed to :meth:`~grgg.model._sampling.Sampler.sample`."""
        return Sampler(self).sample(**kwargs)

    def materialize(self, *, copy: bool = True) -> "GRGG":
        """Materialize a new GRGG model with only the selected nodes.

        Parameters
        ----------
        copy
            Whether to return a deep copy of the model.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = GRGG(100, 2, Similarity(2, jnp.zeros(100)), Complementarity(1, 0))
        >>> submodel = model.nodes[:10].materialize()
        >>> submodel.n_nodes
        10
        >>> submodel.manifold.volume
        100.0
        >>> submodel.layers[0].mu.shape
        (10,)
        """
        if not isinstance(self.module, AbstractGRGG):
            errmsg = "only views of the full GRGG model can be materialized"
            raise TypeError(errmsg)
        if self._i is None:
            return self.module
        model = self.module.deepcopy() if copy else self.module
        layers = [
            layer.replace(
                beta=beta.copy() if copy else beta, mu=mu.copy() if copy else mu
            )
            for layer, beta, mu in zip(model.layers, self.beta, self.mu, strict=False)
        ]
        return model.replace(
            n_nodes=self.n_nodes,
            manifold=model.manifold,
            layers=layers,
        )

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> jnp.ndarray:
        """Get parameter."""
        param = params[name]
        if self._i is not None and not jnp.isscalar(param):
            return param[self._i]
        return param

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[jnp.ndarray]:
        return [self._get_param(p, name) for p in params]


class NodePairView(AbstractModelView):
    """Node pairs view.

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

    _i: int | slice | Sequence[int] | None = eqx.field(
        default=None,
        repr=False,
        kw_only=True,
    )
    _j: int | slice | Sequence[int] | None = eqx.field(
        default=None,
        repr=False,
        kw_only=True,
    )

    def __getitem__(self, args: Any) -> Self:
        if isinstance(args, tuple) and len(args) == 2:
            _i = args
            _j = None
            return replace(self, _i=_i, _j=_j)
        if self._i is None:
            return replace(self, _i=args)
        if self._j is None:
            return replace(self, _j=args)
        errmsg = "too many indices for node pairs"
        raise IndexError(errmsg)

    @property
    def beta(self) -> LazyOuter | list[LazyOuter]:
        """Beta parameter outer product."""
        return self._get_param(self.module.parameters, "beta")

    @property
    def mu(self) -> LazyOuter | list[LazyOuter]:
        """Mu parameter outer product."""
        return self._get_param(self.module.parameters, "mu")

    @property
    def is_active(self) -> bool:
        """Whether the view is active (i.e., has any indices selected)."""
        return self._i is not None and self._j is not None

    def reset(self) -> None:
        """Reset the current node pair selection."""
        self._i = None
        self._j = None

    def probs(self, g: jnp.ndarray) -> jnp.ndarray:
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

        Evaluate the multilayer model probabilities.
        >>> model.pairs[[0, 1]][[0, 2]].probs(1)
        Array([[0.999245 , 0.9996325],
               [0.9969842, 0.5      ]], ...)
        """
        return self._get_probs(self.module.parameters, g)

    @singledispatchmethod
    def _get_param(self, params: Mapping, name: str) -> LazyOuter:
        param = params[name]
        if jnp.isscalar(param):
            param = param / 2
        outer = LazyOuter(param, op=jnp.add)[self._i]
        if self._j is not None:
            return outer[:, self._j]
        return outer[...]

    @_get_param.register
    def _(self, params: Sequence, name: str) -> list[jnp.ndarray]:
        return [self._get_param(p, name) for p in params]

    @singledispatchmethod
    def _get_probs(self, params: Mapping, g: jnp.ndarray) -> jnp.ndarray:
        beta = self._get_param(params, "beta")
        mu = self._get_param(params, "mu")
        return self.module.function(g, beta, mu)

    @_get_probs.register
    def _(self, params: Sequence, g: jnp.ndarray) -> list[jnp.ndarray]:
        beta = jnp.stack([self._get_param(p, "beta") for p in params])
        mu = jnp.stack([self._get_param(p, "mu") for p in params])
        return self.module.function(g, beta, mu)
