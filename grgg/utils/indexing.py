"""This module reverse-engineers Numpy-style indexing in an abstract array-free manner
to allow for lazy evaluation of indexing operations and indexed computations.

Most importantly, all indexing operations are compatible with JAX transformations
like JIT compilation, vectorization, and automatic differentiation.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> import equinox as eqx
>>> from grgg.utils.indexing import IndexExpression
>>> X = jnp.arange(3 * 5 * 6).reshape(3, 5, 6).astype(float)
>>> expr = IndexExpression(X.shape)
>>>
>>> @eqx.filter_jit
... def compute_sum(X, index):
...     expr = IndexExpression(X.shape)
...     return jnp.sum(X[index.coords])
>>>
>>> index = expr.dynamic[:5, [0, 2], 1]
>>> compute_sum(X, index)
Array(222., ...)
>>> grad = jax.grad(compute_sum)(X, index)
>>> grad.shape
(3, 5, 6)
>>> grad
Array([[[0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]], ...)

In general, arbitrary indexing is supported.
>>> X = jnp.arange(3 * 5 * 6).reshape(3, 5, 6)
>>> expr = IndexExpression(X.shape)
>>> expr
IndexExpression(3, 5, 6)
>>> args = expr[:, 2, None, [0, 2], None]  # utility for generating index args
>>> args
(slice(0, 3, 1),
    Array(2, ...),
    None,
    Array([0, 2], ...),
    None)
>>> index = expr.dynamic[args]
>>> index
DynamicIndex(DynamicSlice(0, 3, 1), 2, None, i32[2], None)
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[1, 4]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[..., 2, None, [0, 2]]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[jnp.ix_(jnp.array([1, 3]), jnp.array([0, 2]))]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[X > 10]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[[True, False, True], None, 1, [0, 3]]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[...]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
>>> args = expr[()]
>>> index = expr.dynamic[args]
>>> jnp.array_equal(X[args], X[index.coords]).item()
True
"""
import math
from collections.abc import Sequence
from types import EllipsisType
from typing import Self, cast

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Integers, IntVector
from grgg.abc import AbstractModule
from grgg.utils.misc import format_array

__all__ = ("Shaped", "IndexExpression", "DynamicIndex", "DynamicSlice")


IndexArgT = slice | int | Integers | EllipsisType | None
SliceMapT = dict[int, "DynamicSlice"]
ArrayMapT = dict[int, Integers]
NewaxisMapT = dict[int, None]


class Shaped(AbstractModule):
    """A mixin class for modules that have a shape."""

    shape: eqx.AbstractVar[tuple[int, ...]]

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The total number of elements in the array."""
        return math.prod(self.shape)


class DynamicSlice(AbstractModule):
    """Dynamic slice compatible with JAX transformations.

    Attributes
    ----------
    start
        The starting index of the slice.
    end
        The ending index of the slice.
    step
        The step size of the slice.

    Examples
    --------
    >>> ds = DynamicSlice(2, 10, 2)
    >>> ds
    DynamicSlice(2, 10, 2)
    >>> ds.indices
    Array([2, 4, 6, 8], dtype=int32)
    >>> s = slice(2, 10, 2)
    >>> ds2 = DynamicSlice.from_slice(slice(2, 10, 2))
    >>> ds2
    DynamicSlice(2, 10, 2)
    >>> ds2.equals(ds)
    True
    >>> ds2.to_slice() == s
    True
    """

    start: int = eqx.field(static=True)
    end: int = eqx.field(static=True)
    step: int = eqx.field(static=True)

    def __init__(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
    ) -> None:
        if start is not None and end is None:
            start, end = 0, start
        elif start is None:
            start = 0
        self.start = int(start or 0)
        self.end = int(end)
        self.step = int(step or 1)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}({self.start}, {self.end}, {self.step})"

    @property
    def indices(self) -> IntVector:
        """Get the indices represented by the slice."""
        return jnp.arange(self.start, self.end, self.step)

    @classmethod
    def from_slice(cls, s: slice) -> Self:
        """Create a standard slice object.

        Parameters
        ----------
        s
            The slice object to convert.
        """
        return cls(s.start, s.stop, s.step)

    def to_slice(self) -> slice:
        """Convert to a standard slice object."""
        return slice(self.start, self.end, self.step)


class DynamicIndex(Shaped, Sequence):
    """Dynamic index compatible with JAX transformations.

    Attributes
    ----------
    args
        The indexing arguments.

    Examples
    --------
    >>> from grgg.utils.indexing import DynamicIndex, DynamicSlice
    >>> index = DynamicIndex()
    >>> index
    DynamicIndex()
    >>> index.args
    ()
    >>> DynamicIndex(DynamicSlice(0, 3, 1))
    DynamicIndex(DynamicSlice(0, 3, 1))
    """

    args: tuple[DynamicSlice | Integers | None, ...]

    def __init__(
        self,
        *index_args,
        args: tuple[DynamicSlice | Integers | None, ...] | None = None,
    ) -> None:
        if args is None:
            args = index_args
        elif index_args:
            errmsg = "cannot pass arguments both positionally and as 'args' keyword"
            raise ValueError(errmsg)
        if not isinstance(args, tuple):
            args = (args,)
        self.args = tuple(
            DynamicSlice.from_slice(a)
            if isinstance(a, slice)
            else jnp.asarray(a)
            if not isinstance(a, DynamicSlice | jnp.ndarray | None)
            else a
            for a in args
        )

    def __check_init__(self) -> None:
        for a in self.args:
            if not (
                isinstance(a, DynamicSlice)
                or (isinstance(a, jnp.ndarray) and jnp.issubdtype(a.dtype, jnp.integer))
                or a is None
            ):
                errmsg = f"invalid indexing argument: {a!r}"
                raise TypeError(errmsg)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        args = ", ".join(
            format_array(a) if isinstance(a, jnp.ndarray) else str(a) for a in self.args
        )
        return f"{cn}({args})"

    def __getitem__(self, i: int) -> DynamicSlice | Integers | None:
        return self.args[i]

    def __len__(self) -> int:
        return len(self.args)

    def __add__(self, other: object) -> Self:
        if not isinstance(other, DynamicIndex):
            return NotImplemented
        return DynamicIndex(self.args + other.args)

    def __mul__(self, n: int) -> Self:
        if not isinstance(n, int):
            return NotImplemented
        return DynamicIndex(self.args * n)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the result of the indexing operation
        assuming that each axis is indexed.

        Examples
        --------
        >>> expr = DynamicIndexExpression(3, 5, 6, 4)
        >>> index = expr[:5, [0, 2], 1]
        >>> index.shape
        (3, 2, 4)
        >>> index = expr[1:4, [0, 2], :5, 1]
        >>> index.shape
        (2, 2, 5)
        >>> index = expr[[0, 2], [0, 3]]
        >>> index.shape
        (2, 6, 4)
        >>> index = expr[2, 1, 4, 2]
        >>> index.shape
        ()
        >>> index = expr[2, None, [1, 3]]
        >>> index.shape
        (2, 1, 6, 4)
        >>> index = expr[[True, False], [False, True]]
        >>> index.shape
        (1, 6, 4)
        >>> index = expr[[True, False, False], None]
        >>> index.shape
        (1, 1, 5, 6, 4)
        >>> index = expr[[True, True, False], [1, 3]]
        >>> index.shape
        (2, 6, 4)
        """
        return self.get_shape()

    @property
    def coords(self) -> tuple[Integers, ...]:
        """The coordinate arrays representing the indexing operation."""
        return self.get_coords(resolve_newaxes=False)

    @property
    def has_slice_indexing(self) -> bool:
        """Whether the indexing arguments contain slice indexing."""
        return any(isinstance(a, DynamicSlice) for a in self.args)

    @property
    def has_array_indexing(self) -> bool:
        """Whether the indexing arguments contain array indexing."""
        return any(isinstance(a, jnp.ndarray) for a in self.args)

    @property
    def has_contiguous_array_indexing(self) -> bool:
        """Whether the indexing arguments contain contiguous array indexing.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg.utils.indexing import DynamicIndexExpression
        >>> expr = DynamicIndexExpression(3, 5, 6, 4)
        >>> index = expr[:5, jnp.array([0, 2]), 1]
        >>> index.has_contiguous_array_indexing
        True
        >>> index = expr[jnp.array([0, 2]), slice(2, 5), 1]
        >>> index.has_contiguous_array_indexing
        False
        >>> index = expr[jnp.array([0, 2]), None, jnp.array([1, 3])]
        >>> index.has_contiguous_array_indexing
        False
        >>> index = expr[1:7, None, jnp.array([1, 3])]
        >>> index.has_contiguous_array_indexing
        True
        >>> index = expr[2, [0, 2], 1, 3, None]
        >>> index.has_contiguous_array_indexing
        True
        >>> index = expr[2, 1]
        >>> index.has_contiguous_array_indexing
        True
        >>> index = expr[[0,1,2]]
        >>> index.has_contiguous_array_indexing
        True
        """
        if not self.has_array_indexing:
            return False
        adv_pos = [i for i, a in enumerate(self.args) if isinstance(a, jnp.ndarray)]
        return max(adv_pos) - min(adv_pos) + 1 == len(adv_pos) if adv_pos else False

    @property
    def index_maps(self) -> tuple[SliceMapT, ArrayMapT, NewaxisMapT]:
        """Get the index maps for basic and advanced indexing."""
        slice_map = {}
        array_map = {}
        newaxis_map = {}
        for i, a in enumerate(self.args):
            if a is None:
                newaxis_map[i] = a
            elif isinstance(a, DynamicSlice):
                slice_map[i] = a
            else:
                array_map[i] = a
        return slice_map, array_map, newaxis_map

    def get_coords(
        self, *, resolve_newaxes: bool = False
    ) -> tuple[Integers | None, ...]:
        """Create the coordinate arrays representing the indexing operation.

        Parameters
        ----------
        resolve_newaxes
            If True, new axes (None) are represented as arrays with a single zero value
            and appropriate broadcasted shape. If False, new axes are represented as
            `None`. The first representation is convenient for determining resultant
            shapes, while the second is more useful for actual indexing operations on
            arrays.
        """
        slice_map, array_map, newaxis_map = self.index_maps
        # Broadcast advanced indices
        advanced = tuple(array_map.values()) if array_map else ()
        jnp.broadcast_shapes(*(a.shape for a in advanced))
        for k, a in zip(array_map, advanced, strict=True):
            array_map[k] = a
        # Create a meshgrid of all slice indices
        slice_map = {
            k: v.indices for k, v in slice_map.items() if isinstance(v, DynamicSlice)
        }
        if resolve_newaxes:
            for k in newaxis_map:
                slice_map[k] = jnp.array([0])
        slice_map = dict(sorted(slice_map.items(), key=lambda item: item[0]))
        slice_map = dict(zip(slice_map, jnp.ix_(*slice_map.values()), strict=True))
        if array_map and slice_map:
            # Determine the axis ordering
            adv_pos = (
                0
                if not array_map or not self.has_contiguous_array_indexing
                else min(array_map)
            )
            slice_ndim = list(slice_map.values())[0].ndim if slice_map else 0
            array_ndim = list(array_map.values())[0].ndim if array_map else 0
            # Expand slice indices for broadcasting with advanced indices
            for k, i in slice_map.items():
                if jnp.isscalar(i):
                    continue
                left = (slice(None),) * adv_pos
                expand = (None,) * array_ndim
                slice_map[k] = i[*left, *expand, ...]
            # Expand advanced indices for broadcasting with slice indices
            for k, i in array_map.items():
                left = (None,) * adv_pos
                right = (None,) * (slice_ndim - adv_pos)
                array_map[k] = i[*left, ..., *right]
        joint_mapping = {**slice_map, **array_map}
        if not resolve_newaxes:
            joint_mapping.update(newaxis_map)
        arrays = tuple(joint_mapping[i] for i in range(len(self.args)))
        return arrays

    def get_shape(self, coords: tuple[Integers, ...] | None = None) -> tuple[int, ...]:
        """Get the shape of the result of the indexing operation
        assuming that each axis is indexed.
        """
        if coords is None:
            coords = cast(tuple[Integers, ...], self.get_coords(resolve_newaxes=True))
        return jnp.broadcast_shapes(*(a.shape for a in coords))


class IndexExpression(Shaped):
    """Index expression relative to a target shape.

    Attributes
    ----------
    shape
        JAX-compatible array-based representation of the shape.
    is_dynamic
        Whether the index expression is dynamic (i.e., produces a DynamicIndex)
        or static (i.e., produces a tuple of standard indexing arguments).

    Examples
    --------
    >>> expr = IndexExpression(3, 5, 6)
    >>> expr
    IndexExpression(3, 5, 6)
    >>> expr[..., 2, None, [0, 2]]
    (slice(0, 3, 1),
     Array(2, ...),
     None,
     Array([0, 2], ...))
    >>> expr[..., 2, None, [0, 2], 4]
    (Array(2, ...),
     None,
     Array([0, 2], ...),
     Array(4, ...))
    >>> expr[..., 2, None, ..., [0, 2]]
    Traceback (most recent call last):
        ...
    IndexError: an index can only have a single ellipsis ('...')
    >>> expr[..., 2, None, [0, 2], 4, :]
    Traceback (most recent call last):
        ...
    IndexError: too many indices for shape (3, 5, 6)
    """

    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        shape: int | tuple[int, ...],
        *more_sizes: int,
    ) -> None:
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.shape = shape + more_sizes

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}{self.shape}"

    def __getitem__(
        self, args: IndexArgT | tuple[IndexArgT, ...]
    ) -> tuple[IndexArgT, ...]:
        args = self._resolve_args(args)
        self = self._with_newaxes(args)
        args = self._resolve_slices(args)
        return args

    @property
    def dynamic(self) -> Self:
        """A dynamic version of this index expression."""
        return DynamicIndexExpression(self.shape)

    @property
    def static(self) -> Self:
        """A static version of this index expression."""
        return self

    def _resolve_args(
        self, args: IndexArgT | tuple[IndexArgT, ...]
    ) -> tuple[IndexArgT, ...]:
        if not isinstance(args, tuple):
            args = (args,)
        args = tuple(
            a if isinstance(a, slice | None | EllipsisType) else jnp.asarray(a)
            for a in args
        )
        args = sum(
            (
                a.nonzero()
                if isinstance(a, jnp.ndarray) and jnp.issubdtype(a.dtype, jnp.bool)
                else (a,)
                for a in args
            ),
            start=(),
        )
        num_newaxis = sum(1 for a in args if a is None)
        num_ellipsis = sum(1 for a in args if a is Ellipsis)
        if num_ellipsis > 1:
            errmsg = "an index can only have a single ellipsis ('...')"
            raise IndexError(errmsg)
        if len(args) - num_newaxis - num_ellipsis > self.ndim:
            errmsg = f"too many indices for shape {self.shape}"
            raise IndexError(errmsg)
        ellipsis_index = args.index(Ellipsis) if num_ellipsis == 1 else len(args)
        num_missing = self.ndim - (len(args) - num_newaxis - num_ellipsis)
        args = (
            args[:ellipsis_index]
            + (slice(None),) * num_missing
            + args[ellipsis_index + 1 :]
        )
        return args

    def _with_newaxes(self, args: tuple[IndexArgT, ...]) -> Self:
        new_shape = []
        i = 0
        for a in args:
            if a is None:
                new_shape.append(1)
            else:
                new_shape.append(self.shape[i])
                i += 1
        new_shape = tuple(new_shape) + self.shape[i:]
        return self.replace(shape=new_shape)

    def _resolve_slices(self, args: tuple[IndexArgT, ...]) -> tuple[IndexArgT, ...]:
        return tuple(
            a if not isinstance(a, slice) else slice(*a.indices(s))
            for a, s in zip(args, self.shape, strict=True)
        )


class DynamicIndexExpression(IndexExpression):
    """A dynamic index expression that always produces a DynamicIndex."""

    def __getitem__(self, args: IndexArgT | tuple[IndexArgT, ...]) -> DynamicIndex:
        args = super().__getitem__(args)
        return self._make_dynamic_index(args)

    def _make_dynamic_index(self, args: tuple[IndexArgT, ...]) -> "DynamicIndex":
        _args = tuple(
            DynamicSlice.from_slice(a) if isinstance(a, slice) else a for a in args
        )
        return DynamicIndex(args=_args)

    @property
    def dynamic(self) -> Self:
        """A dynamic version of this index expression."""
        return self

    @property
    def static(self) -> IndexExpression:
        """A static version of this index expression."""
        return IndexExpression(self.shape)
