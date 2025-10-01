import math
from abc import abstractmethod
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self, TypeVar

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Integers, IntVector
from grgg.models.abc import AbstractModelView, AbstractParameter, AbstractParameters
from grgg.statistics import (
    DegreeStatistic,
    QClosureStatistic,
    QClusteringStatistic,
    TClosureStatistic,
    TClusteringStatistic,
)
from grgg.utils.indexing import CartesianCoordinates, IndexArg, Shaped

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
    _coords: CartesianCoordinates = eqx.field(repr=False, init=False)
    _index: IndexArg | tuple[IndexArg, ...] | None = eqx.field(static=False, repr=False)

    motifs_cls: eqx.AbstractClassVar[type[M]]

    def __init__(
        self,
        model: T,
        *,
        _index: IndexArg | tuple[IndexArg, ...] | None = None,
    ) -> None:
        self.model = model
        self._index = self._index_input(_index)
        self._coords = CartesianCoordinates(self.full_shape)

    def __check_init__(self) -> None:
        if (
            self._index is not None
            and isinstance(self._index, tuple)
            and sum(i is not None and i is not Ellipsis for i in self._index)
            > self.full_ndim
        ):
            cn = self.__class__.__name__
            errmsg = (
                f"too many indices for '{cn}': expected up to "
                f"{self.full_ndim}, got {len(self._index)}"
            )
            raise ValueError(errmsg)

    def __getitem__(self, args: Any) -> Self:
        if self._index is None or self._index == self._default_index:
            return self.replace(_index=args)
        cn = self.__class__.__name__
        errmsg = (
            f"'{cn}' can only be indexed once; use the `reset()` method "
            "or `reindex` property to define new indexing from scratch "
            "or call `materialize()` to create a new model with the current selection"
        )
        raise IndexError(errmsg)

    @property
    def _default_index(self) -> IndexArg | tuple[IndexArg, ...]:
        if self.model.is_homogeneous:
            return self._index_input(self._default_homogeneous_index)
        return self._index_input(self._default_heterogeneous_index)

    @property
    def _default_heterogeneous_index(self) -> EllipsisType:
        return ()

    @property
    @abstractmethod
    def _default_homogeneous_index(self) -> int:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the view."""
        return self._coords.s_[self.index]

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
    def index(self) -> IndexArg | tuple[IndexArg, ...]:
        """Current index or default index if 'None' is set."""
        return self._index if self._index is not None else self._default_index

    @property
    def coords(self) -> tuple[Integers, ...]:
        """Coordinates for the selected indices."""
        return self._coords[self.index]

    @property
    def unique_indices(self) -> IntVector:
        """Unique node indices in the view."""
        if not self.is_active or self._index is None:
            return jnp.arange(self.model.n_units)
        uniq = jnp.unique(jnp.concat(tuple(jnp.unique(i) for i in self.coords)))
        return uniq

    @property
    def reindex(self) -> Self:
        """Reindex the view to use Cartesian coordinates."""
        return self.reset()

    def reset(self) -> None:
        """Reset the view."""
        return self.replace(_index=None)

    def equals(self, other: object) -> bool:
        """Check equality with another view."""
        return super().equals(other) and _indices_equal(self._index, other._index)

    def _index_input(
        self, index: IndexArg | tuple[IndexArg, ...] | None
    ) -> IndexArg | tuple[IndexArg, ...] | None:
        """Process input index."""
        if index is None:
            return None
        if index is Ellipsis:
            index = ()
        _index = index if isinstance(index, tuple) else (index,)
        return _index

    def materialize(self, *, copy: bool = False) -> T:
        """Materialize the view into a new model instance.

        Parameters
        ----------
        copy
            Whether to deep copy the model.
        """
        if not self.is_active:
            return self.model if not copy else self.model.copy(deep=True)
        uniq = self.unique_indices
        parameters = self.model.parameters.subset[uniq]
        model = self.model.replace(n_nodes=len(uniq), parameters=parameters)
        if copy:
            model = model.copy(deep=True)
        return model


class AbstractErgmNodeView[T, MV](AbstractErgmView[T, MV]):
    """Abstract base class for ERGM node views."""

    @property
    def _default_homogeneous_index(self) -> int:
        return 0

    @property
    def n_nodes(self) -> int:
        if self._index is None:
            return self.model.n_units
        return max(1, self.shape[0] if self.shape else 0)

    @property
    def full_shape(self) -> tuple[int]:
        return (self.model.n_units,)

    @property
    def sampler(self) -> "S":
        """Sampler for the view."""
        return self.model.sampler_cls(self)

    @property
    def pairs(self) -> "E":
        """View of all node pairs induces by the node view."""
        pairs = self.model.pairs
        if self.index is not None:
            index = self.index
            index = index[0] if isinstance(index, tuple) else index
            if index is Ellipsis:
                return pairs[index]
            if not isinstance(index, slice | int):
                index = jnp.asarray(index)
                if index.ndim > 1:
                    errmsg = (
                        "cannot create pairs view from multi-dimensional node indices"
                    )
                    raise IndexError(errmsg)
                if index.ndim == 1:
                    if jnp.issubdtype(index.dtype, jnp.bool_):
                        errmsg = "cannot create pairs view from boolean node indices"
                        raise IndexError(errmsg)
                    index = jnp.ix_(index, index)
            else:
                index = (index, index)
            if self.n_nodes > 1 and index == (0, 0):
                index = (0, 1)
            return pairs[index]
        return pairs

    def sample(self, *args: Any, **kwargs: Any) -> Any:
        """Sample from the view's sampler."""
        return self.sampler.sample(*args, **kwargs)

    def get_parameter(self, idx: int | str) -> AbstractParameter:
        param = self.model.parameters[idx]
        if self.model.is_homogeneous:
            return jnp.full(self.shape, param.data)
        return param[self.index]

    # Statistics ---------------------------------------------------------------------

    @property
    def degree(self) -> DegreeStatistic:
        """Degree statistic for the nodes in the view."""
        return DegreeStatistic.from_module(self)

    @property
    def tclust(self) -> TClusteringStatistic:
        """Triangle clustering statistic for the nodes in the view."""
        return TClusteringStatistic.from_module(self)

    @property
    def tclosure(self) -> TClosureStatistic:
        """Triangle closure statistic for the nodes in the view."""
        return TClosureStatistic.from_module(self)

    @property
    def qclust(self) -> QClusteringStatistic:
        """Quadrangle clustering statistic for the nodes in the view."""
        return QClusteringStatistic.from_module(self)

    @property
    def qclosure(self) -> QClosureStatistic:
        """Quadrangle closure statistic for the nodes in the view."""
        return QClosureStatistic.from_module(self)


class AbstractErgmNodePairView[T, ME](AbstractErgmView[T, ME]):
    """Abstract base class for node pair views."""

    @property
    def _default_homogeneous_index(self) -> tuple[int, int]:
        return (0, 0) if self.model.n_units <= 1 else (0, 1)

    @property
    def full_shape(self) -> tuple[int, int]:
        n = self.model.n_units
        return (n, n)

    @property
    def n_nodes(self) -> int:
        if self._index is None:
            return self.model.n_units
        return max(self.shape) if self.shape else 1

    @property
    def ij(self) -> tuple[Integers, Integers]:
        """Aligned indices for selected pairs."""
        index = self.index
        args = tuple(a for a in (index if index is not None else ()) if a is not None)
        coords = self._coords[args]
        if len(coords) < 2:
            coords = self.index
        return coords

    @property
    def nodes(self) -> "V":
        """View of all nodes induces by the node pair view."""
        uniq = self.unique_indices
        return self.model.nodes[uniq]

    def get_parameter(self, idx: int | str) -> Any:
        """Get a model parameter by index or name."""
        param = self.model.parameters[idx]
        if self.model.is_homogeneous:
            return param.data
        param = param.outer
        if self._index is None:
            return param[...]
        return param[self.index][...]


# Internals --------------------------------------------------------------------------


def _indices_equal(
    a: IndexArg | tuple[IndexArg, ...] | None,
    b: IndexArg | tuple[IndexArg, ...] | None,
) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, jnp.ndarray):
        if isinstance(b, jnp.ndarray):
            return jnp.array_equal(a, b)
        return False
    if isinstance(a, tuple):
        if isinstance(b, tuple):
            if len(a) != len(b):
                return False
            return all(_indices_equal(ai, bi) for ai, bi in zip(a, b, strict=True))
        return False
    return a == b
