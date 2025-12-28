import math
from abc import abstractmethod
from functools import wraps
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import Integers, IntVector
from grgg.statistics import (
    Degree,
    QClosure,
    QClustering,
    QStatistics,
    StructuralComplementarity,
    StructuralSimilarity,
    TClosure,
    TClustering,
    TStatistics,
)
from grgg.utils.dispatch import dispatch
from grgg.utils.indexing import (
    DynamicIndex,
    DynamicIndexExpression,
    IndexArgT,
    Shaped,
)
from grgg.utils.misc import cartesian_product
from grgg.utils.random import RandomGenerator

from ..model import AbstractModelView, AbstractParameter
from .motifs import (
    AbstractErgmMotifs,
    ErgmNodeMotifs,
    ErgmNodePairMotifs,
)

if TYPE_CHECKING:
    from .model import AbstractErgm

    T = TypeVar("T", bound=AbstractErgm)
    NV = TypeVar("NV", bound="AbstractErgmNodeView[T]")
    PV = TypeVar("PV", bound="AbstractErgmNodePairView[T]")

__all__ = ("AbstractErgmView", "AbstractErgmNodeView", "AbstractErgmNodePairView")


class AbstractErgmView[T](AbstractModelView[T], Shaped):
    """Abstract base class for ERGM views."""

    _index: DynamicIndex | None = eqx.field(repr=False)

    motifs_cls: eqx.AbstractClassVar[AbstractErgmMotifs[T]]

    def __init__(
        self,
        model: T,
        *,
        _index: DynamicIndex | None = None,
    ) -> None:
        self.model = model
        if _index is not None and not isinstance(_index, DynamicIndex):
            _index = self.index_expr[_index]
        self._index = _index

    def __check_init__(self) -> None:
        if self._index is not None and (ndim := self._index.ndim) > self.full_ndim:
            cn = self.__class__.__name__
            errmsg = (
                f"too many indices for '{cn}': expected up to "
                f"{self.full_ndim}, got {ndim}"
            )
            raise ValueError(errmsg)

    def __getitem__(self, args: Any) -> Self:
        if not self.is_active or self._index.equals(
            DynamicIndex(self._default_index_args)
        ):
            return self.replace(_index=self.index_expr[args])
        cn = self.__class__.__name__
        errmsg = (
            f"'{cn}' can only be indexed once; use the `reset()` method "
            "or `reindex` property to define new indexing from scratch "
            "or call `materialize()` to create a new model with the current selection"
        )
        raise IndexError(errmsg)

    @property
    def index(self) -> DynamicIndex:
        """Current index of the view."""
        if self._index is None:
            expr = DynamicIndexExpression(self.full_shape)
            return expr[self._default_index_args]
        return self._index

    @property
    def _default_index_args(self) -> IndexArgT | tuple[IndexArgT, ...]:
        if self.model.is_homogeneous:
            return self._default_homogeneous_index_args
        return self._default_heterogeneous_index_args

    @property
    def _default_heterogeneous_index_args(self) -> EllipsisType:
        return ()

    @property
    @abstractmethod
    def _default_homogeneous_index_args(self) -> int:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the view."""
        return self.index.shape

    @property
    def is_active(self) -> bool:
        """Whether the view is active (i.e., has any indices selected)."""
        return self._index is not None

    @property
    @abstractmethod
    def n_nodes(self) -> int:
        """Number of nodes in the view."""

    @property
    @abstractmethod
    def full_shape(self) -> tuple[int, ...]:
        """Shape of the full model view."""

    @property
    def full_ndim(self) -> int:
        return len(self.full_shape)

    @property
    def full_size(self) -> int:
        return math.prod(self.full_shape)

    @property
    def coords(self) -> tuple[Integers, ...]:
        """Coordinates for the selected indices."""
        return self.index.coords

    @property
    @abstractmethod
    def unique_indices(self) -> Integers:
        """Unique indices in the view."""

    @property
    def reindex(self) -> Self:
        """Reindex the view to use Cartesian coordinates."""
        return self.reset()

    @property
    def index_expr(self) -> DynamicIndexExpression:
        """Index expression for the view."""
        return DynamicIndexExpression(self.full_shape)

    @property
    def motifs(self) -> AbstractErgmMotifs[T]:
        """Motif statistics for the view."""
        return self.motifs_cls(self)

    def reset(self) -> None:
        """Reset the view."""
        return self.replace(_index=None)

    @abstractmethod
    def materialize(self, indices: Integers | None = None, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        if not self.is_active:
            return self.model if not copy else self.model.copy(deep=True)
        if indices is None:
            errmsg = "`indices` must be provided to materialize a view"
            raise ValueError(errmsg)
        parameters = self.model.parameters.replace(
            **{
                name: param[indices]
                for name, param in self.model.parameters.mapping.items()
            }
        )
        model = self.model.replace(n_nodes=len(indices), parameters=parameters)
        if copy:
            model = model.copy(deep=True)
        return model

    @dispatch.abstract
    def _get_statistic(self, name: str) -> Any:
        """Get the statistic by name for the given model."""


class AbstractErgmNodeView[T](AbstractErgmView[T]):
    """Abstract base class for ERGM node views."""

    motifs_cls: ClassVar[ErgmNodeMotifs] = ErgmNodeMotifs  # type: ignore

    @property
    def _default_homogeneous_index_args(self) -> int:
        return 0

    @property
    def n_nodes(self) -> int:
        if not self.is_active:
            return self.model.n_units
        return len(self.unique_indices)

    @property
    def full_shape(self) -> tuple[int]:
        return (self.model.n_units,)

    @property
    def unique_indices(self) -> Integers:
        indices = jnp.concat([a.flatten() for a in self.coords], axis=0)
        return jnp.unique(indices)

    @property
    def pairs(self) -> "PV":
        """View of all node pairs induces by the node view."""
        pairs = self.model.pairs
        if not self.is_active:
            return pairs
        coords = jnp.ix_(*(self.coords * 2))
        return pairs[coords]

    def get_parameter(self, idx: str) -> AbstractParameter:
        idx = self.model.parameters.names.index(idx)
        param = self.model.parameters[idx]
        if param.is_homogeneous or self._index is None:
            return param
        return param[self.index.coords]

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        indices = self.unique_indices
        return super().materialize(indices, copy=copy)

    def sample(
        self,
        n: int,
        *,
        replace: bool = False,
        rng: RandomGenerator | None = None,
        **kwargs: Any,
    ) -> Self:
        """Randomly sample nodes from the view.

        Parameters
        ----------
        n
            Number of nodes to sample.
        replace
            Whether to sample with replacement.
        rng
            Random number generator to use.
        **kwargs
            Additional keyword arguments passed to :func:`jax.random.choice`.

        Examples
        --------
        >>> from grgg import RandomGraph, RandomGenerator
        >>> rng = RandomGenerator(177)
        >>> n = 100
        >>> model = RandomGraph(n)
        >>> view = model.nodes.sample(10, rng=rng)
        >>> view.n_nodes
        10
        >>> view.sample(20, replace=True, rng=rng)
        Traceback (most recent call last):
        ...
        IndexError: '...NodeView' can only be indexed once
        >>> view = model.nodes.sample(200, replace=True, rng=rng)
        >>> view.n_nodes <= 100
        True
        >>> len(view.coords[0])
        200
        >>> len(view.degree())
        200
        """
        if isinstance(rng, RandomGenerator):
            key = rng.child.key
        else:
            key = RandomGenerator.make_key(rng)
        indices = jax.random.choice(key, self.n_nodes, (n,), replace=replace, **kwargs)
        return self[indices]

    # Statistics ---------------------------------------------------------------------

    @property
    @wraps(Degree.__init__)
    def degree(self) -> Degree:
        """Degree statistic for the nodes in the view."""
        return self._get_statistic("degree")

    @property
    @wraps(TClustering.__init__)
    def tclust(self) -> TClustering:
        """Triangle clustering statistic for the nodes in the view."""
        return self._get_statistic("tclust")

    @property
    @wraps(TClosure.__init__)
    def tclosure(self) -> TClosure:
        """Triangle closure statistic for the nodes in the view."""
        return self._get_statistic("tclosure")

    @property
    @wraps(StructuralSimilarity.__init__)
    def similarity(self) -> StructuralSimilarity:
        """Structural similarity statistic for the nodes in the view."""
        return self._get_statistic("similarity")

    @property
    @wraps(QClustering.__init__)
    def qclust(self) -> QClustering:
        """Quadrangle clustering statistic for the nodes in the view."""
        return self._get_statistic("qclust")

    @property
    @wraps(QClosure.__init__)
    def qclosure(self) -> QClosure:
        """Quadrangle closure statistic for the nodes in the view."""
        return self._get_statistic("qclosure")

    @property
    @wraps(StructuralComplementarity.__init__)
    def complementarity(self) -> StructuralComplementarity:
        """Complementarity statistic for the nodes in the view."""
        return self._get_statistic("complementarity")

    @property
    @wraps(TStatistics.__init__)
    def tstats(self) -> jnp.ndarray:
        """Triangle statistics for the nodes in the view."""
        return self._get_statistic("tstats")

    @property
    @wraps(QStatistics.__init__)
    def qstats(self) -> jnp.ndarray:
        """Quadrangle statistics for the nodes in the view."""
        return self._get_statistic("qstats")


class AbstractErgmNodePairView[T](AbstractErgmView[T]):
    """Abstract base class for node pair views."""

    motifs_cls: ClassVar[ErgmNodePairMotifs] = ErgmNodePairMotifs  # type: ignore

    @property
    def _default_homogeneous_index_args(self) -> tuple[int, int]:
        return (0, 0) if self.model.n_units <= 1 else (0, 1)

    @property
    def full_shape(self) -> tuple[int, int]:
        n = self.model.n_units
        return (n, n)

    @property
    def n_nodes(self) -> int:
        return len(self.node_indices)

    @property
    def n_pairs(self) -> int:
        return len(self.unique_indices)

    @property
    def unique_indices(self) -> Integers:
        if not self.is_active:
            nodes = self.node_indices
            indices = jnp.ix_(nodes, nodes)
        else:
            indices = jnp.broadcast_arrays(*self.coords)
        indices = cartesian_product([i.squeeze() for i in indices])
        return jnp.unique(indices, axis=0)

    @property
    def node_indices(self) -> IntVector:
        """Unique node indices in the view."""
        if not self.is_active:
            return jnp.arange(self.model.n_units)
        return jnp.unique(self.unique_indices)

    @property
    def nodes(self) -> "NV":
        """View of all nodes induces by the node pair view."""
        return self.model.nodes[self.node_indices]

    def get_parameter(self, idx: int | str) -> AbstractParameter:
        """Get a model parameter by index or name."""
        if isinstance(idx, str):
            idx = self.model.parameters.names.index(idx)
        param = self.model.parameters[idx]
        if param.is_homogeneous:
            return param.data
        i, j = self.coords if self.is_active else self[...].coords
        return param[i] + param[j]

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        indices = self.node_indices
        return super().materialize(indices, copy=copy)
