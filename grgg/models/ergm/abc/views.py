import math
from abc import abstractmethod
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Integers, IntVector
from grgg.models.abc import AbstractModelView, AbstractParameter, AbstractParameters
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
from grgg.utils.indexing import (
    DynamicIndex,
    DynamicIndexExpression,
    IndexArgT,
    Shaped,
)
from grgg.utils.lazy import LazyOuter

from .motifs import (
    AbstractErgmMotifs,
    AbstractErgmNodeMotifs,
    AbstractErgmNodePairMotifs,
)
from .sampling import AbstractErgmSampler

if TYPE_CHECKING:
    from .models import AbstractErgm

    P = TypeVar("P", bound=AbstractParameters)
    V = TypeVar("V", bound="AbstractErgmNodeView")
    E = TypeVar("E", bound="AbstractErgmNodePairView")
    S = TypeVar("S", bound=AbstractErgmSampler)
    T = TypeVar("T", bound=AbstractErgm[P, V, E, S])
    M = TypeVar("M", bound="AbstractErgmMotifs")
    MV = TypeVar("M", bound="AbstractErgmNodeMotifs")
    ME = TypeVar("N", bound="AbstractErgmNodePairMotifs")

__all__ = ("AbstractErgmView", "AbstractErgmNodeView", "AbstractErgmNodePairView")


class AbstractErgmView[T, M](AbstractModelView[T], Shaped):
    """Abstract base class for ERGM views."""

    model: eqx.AbstractVar[T]
    _index: DynamicIndex | None = eqx.field(repr=False)

    motifs_cls: eqx.AbstractClassVar[type[M]]

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
    def motifs(self) -> M:
        """Motif statistics for the view."""
        return self.motifs_cls(self)

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

    def reset(self) -> None:
        """Reset the view."""
        return self.replace(_index=None)

    def _equals(self, other: object) -> bool:
        """Check equality with another view."""
        return (
            super()._equals(other)
            and self.model.equals(other.model)
            and self.index.equals(other.index)
        )

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
        parameters = self.model.parameters.subset[indices]
        model = self.model.replace(n_nodes=len(indices), parameters=parameters)
        if copy:
            model = model.copy(deep=True)
        return model


class AbstractErgmNodeView[T, MV](AbstractErgmView[T, MV]):
    """Abstract base class for ERGM node views."""

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
    def sampler(self) -> "S":
        """Sampler for the view."""
        return self.model.sampler_cls(self)

    @property
    def pairs(self) -> "E":
        """View of all node pairs induces by the node view."""
        pairs = self.model.pairs
        if not self.is_active:
            return pairs
        coords = jnp.ix_(*(self.coords * 2))
        return pairs[coords]

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the view's sampler."""
        return self.sampler.sample(*args, **kwargs)

    def get_parameter(self, idx: int | str) -> AbstractParameter:
        param = self.model.parameters[idx]
        if self.model.is_homogeneous:
            return param
        return param[self._index]

    # Statistics ---------------------------------------------------------------------

    @property
    def degree(self) -> Degree:
        """Degree statistic for the nodes in the view."""
        return Degree.from_module(self)

    @property
    def tclust(self) -> TClustering:
        """Triangle clustering statistic for the nodes in the view."""
        return TClustering.from_module(self)

    @property
    def tclosure(self) -> TClosure:
        """Triangle closure statistic for the nodes in the view."""
        return TClosure.from_module(self)

    @property
    def similarity(self) -> StructuralSimilarity:
        """Structural similarity statistic for the nodes in the view."""
        return StructuralSimilarity.from_module(self)

    @property
    def qclust(self) -> QClustering:
        """Quadrangle clustering statistic for the nodes in the view."""
        return QClustering.from_module(self)

    @property
    def qclosure(self) -> QClosure:
        """Quadrangle closure statistic for the nodes in the view."""
        return QClosure.from_module(self)

    @property
    def complementarity(self) -> StructuralComplementarity:
        """Complementarity statistic for the nodes in the view."""
        return StructuralComplementarity.from_module(self)

    @property
    def tstats(self) -> jnp.ndarray:
        """Triangle statistics for the nodes in the view."""
        return TStatistics.from_module(self)

    @property
    def qstats(self) -> jnp.ndarray:
        """Quadrangle statistics for the nodes in the view."""
        return QStatistics.from_module(self)

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        indices = self.unique_indices
        return super().materialize(indices, copy=copy)


class AbstractErgmNodePairView[T, ME](AbstractErgmView[T, ME]):
    """Abstract base class for node pair views."""

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
        indices = jnp.stack(indices, axis=-1).reshape(-1, 2)
        return jnp.unique(indices, axis=1)

    @property
    def node_indices(self) -> IntVector:
        """Unique node indices in the view."""
        if not self.is_active:
            return jnp.arange(self.model.n_units)
        return jnp.unique(self.unique_indices)

    @property
    def nodes(self) -> "V":
        """View of all nodes induces by the node pair view."""
        return self.model.nodes[self.node_indices]

    def get_parameter(self, idx: int | str) -> jnp.ndarray | LazyOuter:
        """Get a model parameter by index or name."""
        param = self.model.parameters[idx]
        if self.model.is_homogeneous:
            return param.data
        param = param.outer
        if not self.is_active:
            return param[...]
        return param[self.coords][...]

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        indices = self.node_indices
        return super().materialize(indices, copy=copy)
