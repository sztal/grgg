import math
from abc import abstractmethod
from functools import wraps
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Integers, IntVector
from grgg.models.abc import AbstractModelView
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
from grgg.utils.misc import cartesian_product

from .motifs import (
    AbstractErgmMotifs,
    AbstractErgmNodeMotifs,
    AbstractErgmNodePairMotifs,
)
from .sampling import AbstractErgmSampler

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
        parameters = {
            name: getattr(self.model, name)[indices]
            for name in self.model.Parameters._fields
        }
        model = self.model.replace(n_nodes=len(indices), **parameters)
        if copy:
            model = model.copy(deep=True)
        return model


class AbstractErgmNodeView[T](AbstractErgmView[T]):
    """Abstract base class for ERGM node views."""

    degree_cls: eqx.AbstractClassVar[type[Degree]]
    tclust_cls: eqx.AbstractClassVar[type[TClustering]]
    tclosure_cls: eqx.AbstractClassVar[type[TClosure]]
    similarity_cls: eqx.AbstractClassVar[type[StructuralSimilarity]]
    qclust_cls: eqx.AbstractClassVar[type[QClustering]]
    qclosure_cls: eqx.AbstractClassVar[type[QClosure]]
    complementarity_cls: eqx.AbstractClassVar[type[StructuralComplementarity]]
    tstats_cls: eqx.AbstractClassVar[type[TStatistics]]
    qstats_cls: eqx.AbstractClassVar[type[QStatistics]]

    motifs_cls: eqx.AbstractClassVar[AbstractErgmNodeMotifs[T]]
    sampler_cls: eqx.AbstractClassVar[type[AbstractErgmSampler[T]]]

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

    @property
    def sampler(self) -> AbstractErgmSampler[T]:
        """Sampler for the view."""
        return self.sampler_cls(self)

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the view's sampler."""
        return self.sampler.sample(*args, **kwargs)

    def get_parameter(self, idx: str) -> jnp.ndarray:
        param = self.model.parameters[idx]
        if self.model.is_homogeneous:
            return param.data
        return param[self.index].data

    # Statistics ---------------------------------------------------------------------

    @property
    @wraps(Degree.__init__)
    def degree(self) -> Degree:
        """Degree statistic for the nodes in the view."""
        return self.degree_cls(self)

    def edge_density(self, *args: Any, **kwargs: Any) -> float:
        """Expected edge density of the model."""
        return self.degree(*args, **kwargs).mean() / (self.n_nodes - 1)

    @property
    @wraps(TClustering.__init__)
    def tclust(self) -> TClustering:
        """Triangle clustering statistic for the nodes in the view."""
        return self.tclust_cls(self)

    @property
    @wraps(TClosure.__init__)
    def tclosure(self) -> TClosure:
        """Triangle closure statistic for the nodes in the view."""
        return self.tclosure_cls(self)

    @property
    @wraps(StructuralSimilarity.__init__)
    def similarity(self) -> StructuralSimilarity:
        """Structural similarity statistic for the nodes in the view."""
        return self.similarity_cls(self)

    @property
    @wraps(QClustering.__init__)
    def qclust(self) -> QClustering:
        """Quadrangle clustering statistic for the nodes in the view."""
        return self.qclust_cls(self)

    @property
    @wraps(QClosure.__init__)
    def qclosure(self) -> QClosure:
        """Quadrangle closure statistic for the nodes in the view."""
        return self.qclosure_cls(self)

    @property
    @wraps(StructuralComplementarity.__init__)
    def complementarity(self) -> StructuralComplementarity:
        """Complementarity statistic for the nodes in the view."""
        return self.complementarity_cls(self)

    @property
    @wraps(TStatistics.__init__)
    def tstats(self) -> jnp.ndarray:
        """Triangle statistics for the nodes in the view."""
        return self.tstats_cls(self)

    @property
    @wraps(QStatistics.__init__)
    def qstats(self) -> jnp.ndarray:
        """Quadrangle statistics for the nodes in the view."""
        return self.qstats_cls(self)

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        indices = self.unique_indices
        return super().materialize(indices, copy=copy)


class AbstractErgmNodePairView[T](AbstractErgmView[T]):
    """Abstract base class for node pair views."""

    motifs_cls: eqx.AbstractClassVar[AbstractErgmNodePairMotifs[T]]

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

    def get_parameter(self, idx: int | str) -> jnp.ndarray:
        """Get a model parameter by index or name."""
        if isinstance(idx, str):
            idx = self.model.Parameters._fields.index(idx)
        param = self.model.parameters[idx]
        if self.model.is_homogeneous:
            return param
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
