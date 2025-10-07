"""This module reverse-engineers Numpy-style indexing in an abstract array-free manner
to allow for lazy evaluation of indexing operations and indexed computations.

Most importantly, all indexing operations are compatible with JAX transformations
like JIT compilation, vectorization, and automatic differentiation.

Examples
--------
>>> import jax
>>> import jax.numpy as jnp
>>> X = jnp.arange(3 * 5 * 6).reshape(3, 5, 6).astype(float)
>>> expr = DynamicIndexExpression(X.shape)
>>>
>>> @jax.jit
... def compute_sum(X, index):
...     return jnp.sum(X[index.coords])
>>>
>>> # Utility for generating indexing arguments
>>> args = expr.i_[1:2, [True, False, False, True, True], None, 3]
>>> compute_sum(X, expr[args])
Array(141., ...)
>>> grad = jax.grad(compute_sum)(X, expr[args])
>>> grad.shape
(3, 5, 6)
>>> grad
Array([[[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]],
       [[0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 0.]],
       [[0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0.]]], ...)
"""
import math
from types import EllipsisType
from typing import Self

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Integer, Integers, IntVector
from grgg.abc import AbstractModule

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


class IndexExpression(Shaped):
    """Index expression relative to a target shape.

    Attributes
    ----------
    shape
        The shape of the target array-like object.

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

    def __init__(self, shape: int | tuple[int, ...], *more_sizes: int) -> None:
        if not isinstance(shape, tuple):
            shape = (shape,)
        if more_sizes:
            shape = shape + more_sizes
        self.shape = tuple(int(s) for s in shape)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}{self.shape}"

    def __getitem__(
        self, args: IndexArgT | tuple[IndexArgT, ...]
    ) -> tuple[IndexArgT, ...]:
        return self._resolve_args(args)

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.shape == other.shape

    def _resolve_args(
        self, args: IndexArgT | tuple[IndexArgT, ...]
    ) -> tuple[IndexArgT, ...]:
        if not isinstance(args, tuple):
            args = (args,)
        num_newaxis = sum(1 for a in args if a is None)
        num_ellipsis = sum(1 for a in args if a is Ellipsis)
        if num_ellipsis > 1:
            errmsg = "an index can only have a single ellipsis ('...')"
            raise IndexError(errmsg)
        if len(args) - num_newaxis - num_ellipsis > self.ndim:
            errmsg = f"too many indices for shape {self.shape}"
            raise IndexError(errmsg)
        if num_ellipsis == 1:
            ellipsis_index = args.index(Ellipsis)
            num_missing = self.ndim - (len(args) - num_newaxis - 1)
            args = (
                args[:ellipsis_index]
                + (slice(None),) * num_missing
                + args[ellipsis_index + 1 :]
            )
        args = tuple(a if isinstance(a, slice | None) else jnp.asarray(a) for a in args)
        _args = list(args)
        i = 0
        for a in _args:
            if isinstance(a, slice):
                start, stop, step = a.indices(self.shape[i])
                _args[i] = slice(start, stop, step)
            if a is not None:
                i += 1
        return tuple(_args)


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

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

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


class DynamicIndexing(Shaped):
    """Dynamic indexing arguments compatible with JAX transformations.

    Attributes
    ----------
    args
        The indexing arguments.
    """

    args: tuple[DynamicSlice | Integers | None, ...]

    def __init__(
        self,
        args: IndexArgT | tuple[IndexArgT, ...],
        *more_args: IndexArgT,
    ) -> None:
        if not isinstance(args, tuple):
            args = (args,)
        args = args + more_args
        _args = tuple(
            a
            if isinstance(a, DynamicSlice)
            else DynamicSlice.from_slice(a)
            if isinstance(a, slice)
            else jnp.asarray(a)
            if a is not None
            else a
            for a in args
        )
        self.args = sum(
            (
                a.nonzero()
                if isinstance(a, jnp.ndarray) and jnp.issubdtype(a.dtype, jnp.bool)
                else (a,)
                for a in _args
            ),
            start=(),
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
        return f"{self.__class__.__name__}{self.shape}"

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and len(self) == len(other)
            and all(
                jnp.array_equal(a, b)
                if isinstance(a, jnp.ndarray)
                else a.equals(b)
                if isinstance(a, DynamicSlice)
                else a == b
                for a, b in zip(self.args, other.args, strict=True)
            )
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the result of the indexing operation
        assuming that each axis is indexed.

        Examples
        --------
        >>> index = DynamicIndexing(slice(5), jnp.array([0, 2]), 1)
        >>> index.shape
        (5, 2)
        >>> index = DynamicIndexing(slice(1, 4), jnp.array([0, 2]), slice(5), 1)
        >>> index.shape
        (2, 3, 5)
        >>> index = DynamicIndexing([0, 2], [0, 3])
        >>> index.shape
        (2,)
        >>> index = DynamicIndexing(2, 1)
        >>> index.shape
        ()
        >>> index = DynamicIndexing(2, None, jnp.array([1, 3]))
        >>> index.shape
        (2, 1)
        >>> index = DynamicIndexing([[True, False], [False, True]])
        >>> index.shape
        (2,)
        >>> index = DynamicIndexing([True, False, False], None)
        >>> index.shape
        (1, 1)
        >>> index = DynamicIndexing([True, True, False], [1, 3])
        >>> index.shape
        (2,)
        """
        coords = self.get_coords(newaxes_as_arrays=True)
        return jnp.broadcast_shapes(*(a.shape for a in coords))

    @property
    def coords(self) -> tuple[Integers, ...]:
        """The coordinate arrays representing the indexing operation."""
        return self.get_coords(newaxes_as_arrays=False)

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
        >>> index = DynamicIndexing(slice(5), jnp.array([0, 2]), 1)
        >>> index.has_contiguous_array_indexing
        True
        >>> index = DynamicIndexing(jnp.array([0, 2]), slice(2, 5), 1)
        >>> index.has_contiguous_array_indexing
        False
        >>> index = DynamicIndexing(jnp.array([0, 2]), None, jnp.array([1, 3]))
        >>> index.has_contiguous_array_indexing
        False
        >>> index = DynamicIndexing(slice(7), None, jnp.array([1, 3]))
        >>> index.has_contiguous_array_indexing
        True
        >>> index = DynamicIndexing(2, [0, 2], 1, 3, None)
        >>> index.has_contiguous_array_indexing
        True
        >>> index = DynamicIndexing(2, 1)
        >>> index.has_contiguous_array_indexing
        True
        >>> index = DynamicIndexing([0,1,2])
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
        self, *, newaxes_as_arrays: bool = False
    ) -> tuple[Integers | None, ...]:
        """Create the coordinate arrays representing the indexing operation.

        Parameters
        ----------
        newaxes_as_arrays
            If True, new axes (None) are represented as arrays with a single zero value
            and appropriate broadcasted shape. If False, new axes are represented as
            `None`. The first representation is convenient for determining resultant
            shapes, while the second is more useful for actual indexing operations on
            arrays.
        """
        slice_map, array_map, newaxis_map = self.index_maps
        # Broadcast advanced indices
        advanced = jnp.broadcast_arrays(*array_map.values()) if array_map else ()
        for k, a in zip(array_map, advanced, strict=True):
            array_map[k] = a
        # Create a meshgrid of all slice indices
        slice_map = {
            k: v.indices for k, v in slice_map.items() if isinstance(v, DynamicSlice)
        }
        if newaxes_as_arrays:
            for k in newaxis_map:
                slice_map[k] = jnp.array([0])
        slice_map = dict(zip(slice_map, jnp.ix_(*slice_map.values()), strict=True))
        # Determine the axis ordering
        adv_pos = (
            0
            if not array_map or not self.has_contiguous_array_indexing
            else min(array_map)
        )
        if array_map:
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
        if not newaxes_as_arrays:
            joint_mapping.update(newaxis_map)
        arrays = tuple(joint_mapping[i] for i in range(len(self.args)))
        return arrays


class DynamicShape(Shaped):
    """Dynamic shape compatible with JAX transformations.

    Attributes
    ----------
    sizes
        The sizes of each dimension.
    """

    sizes: IntVector = eqx.field(converter=jnp.asarray)

    def __init__(self, sizes: IntVector | Integer, *more_sizes: Integer) -> None:
        if not isinstance(sizes, jnp.ndarray):
            sizes = jnp.atleast_1d(jnp.asarray(sizes))
        if more_sizes:
            sizes = jnp.concatenate((sizes, jnp.asarray(more_sizes)))
        self.sizes = sizes

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}{self.shape}"

    def __check_init__(self) -> None:
        if self.sizes.ndim != 1:
            errmsg = f"Expected 1D array, got {self.sizes.ndim}D array"
            raise ValueError(errmsg)
        if not jnp.issubdtype(self.sizes.dtype, jnp.integer):
            errmsg = f"Expected integer array, got {self.sizes.dtype}"
            raise TypeError(errmsg)
        if jnp.any(self.sizes < 0):
            errmsg = f"Expected non-negative sizes, got {self.sizes}"
            raise ValueError(errmsg)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape as a tuple of integers."""
        return tuple(int(s) for s in self.sizes)

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and jnp.array_equal(self.sizes, other.sizes)


class DynamicIndexExpression(Shaped):
    """Dynamic index expression compatible with JAX transformations.

    Instances can be indexed using Numpy-style indexing to produce
    :class:`DynamicIndexArgs` objects with arguments resolved for the target shape.

    Examples
    --------
    >>> expr = DynamicIndexExpression(3, 4, 5)
    >>> expr
    DynamicIndexExpression(3, 4, 5)
    >>> X = jnp.arange(3 * 5 * 6).reshape(3, 5, 6)
    >>> expr = DynamicIndexExpression(X.shape)
    >>> args = expr.i_[:, 2, None, [0, 2], None]  # utility for generating index args
    >>> args
    (slice(0, 3, 1),
     Array(2, ...),
     None,
     Array([0, 2], ...),
     None)
    >>> index = expr[args]
    >>> index
    DynamicIndexing(2, 3, 1, 1)
    >>> jnp.array_equal(X[args], X[index.coords]).item()
    True
    >>> args = expr.i_[1, 4]
    >>> jnp.array_equal(X[args], X[expr[args].coords]).item()
    True
    >>> args = expr.i_[..., 2, None, [0, 2]]
    >>> jnp.array_equal(X[args], X[expr[args].coords]).item()
    True
    >>> args = expr.i_[jnp.ix_(jnp.array([1, 3]), jnp.array([0, 2]))]
    >>> jnp.array_equal(X[args], X[expr[args].coords]).item()
    True
    >>> args = expr.i_[X > 10]
    >>> jnp.array_equal(X[args], X[expr[args].coords]).item()
    True
    >>> args = expr.i_[[True, False, True], None, 1, [0, 3]]
    >>> jnp.array_equal(X[args], X[expr[args].coords]).item()
    True
    """

    _shape: DynamicShape

    def __init__(self, _shape: IntVector | Integer, *more_sizes: Integer) -> None:
        if isinstance(_shape, DynamicShape):
            if more_sizes:
                _shape = DynamicShape(_shape.sizes, *more_sizes)
        else:
            _shape = DynamicShape(_shape, *more_sizes)
        self._shape = _shape

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        return f"{cn}{self.shape}"

    def __getitem__(self, args: IndexArgT | tuple[IndexArgT, ...]) -> DynamicIndexing:
        args = self.i_[args]
        self = self._with_newaxes(args)
        index = self._make_index(args)
        return index

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the target array-like object."""
        return self._shape.shape

    @property
    def i_(self) -> IndexExpression:
        """An index expression for the target shape."""
        return IndexExpression(self.shape)

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self._shape.equals(other._shape)

    def _with_newaxes(self, args: tuple[IndexArgT, ...]) -> Self:
        new_shape = []
        i = 0
        for a in args:
            if a is None:
                new_shape.append(1)
            else:
                new_shape.append(self.shape[i])
                i += 1
        return self.__class__(new_shape)

    def _make_index(self, args: tuple[IndexArgT, ...]) -> DynamicIndexing:
        index = []
        for a in args:
            if a is None:
                index.append(a)
            elif isinstance(a, slice):
                a = DynamicSlice(*a.indices(self.shape[len(index)]))
                index.append(a)
            else:
                index.append(a)
        return DynamicIndexing(tuple(index))
