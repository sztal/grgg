"""This module reverse-engineers Numpy-style indexing in an abstract array-free context
to allow for lazy evaluation of indexing operations and indexed computations.
"""
import math
from collections.abc import Sequence
from types import EllipsisType
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax import lax
import numpy as np
from grgg.abc import AbstractModule

__all__ = ("MultiIndexRavel", "IndexableShape", "CartesianCoordinates")

IndexArg = int | slice | EllipsisType | Sequence[int] | jnp.ndarray | np.ndarray


class Shaped(AbstractModule):
    """An object with a known shape."""

    shape: eqx.AbstractVar[tuple[int, ...]]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.shape}"

    @property
    def ndim(self) -> int:
        """Number of dimensions of the index expression."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements in the index expression."""
        return math.prod(self.shape) if self.shape else 1


class ShapedIndexExpression(Shaped):
    """An index expression with a known shape.

    Attributes
    ----------
    shape
        Shape of the index expression.
    """

    shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, *shape: int) -> None:
        shape = sum(((s,) if not isinstance(s, tuple) else s for s in shape), start=())
        shape = tuple(int(s) for s in shape)
        self.shape = shape

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.shape == other.shape

    def __getitem__(
        self,
        args: IndexArg | tuple[IndexArg, ...],
    ) -> tuple[IndexArg, ...]:
        # ruff: noqa
        if not isinstance(args, tuple):
            args = (args,)
        args = self._handle_ellipses(args)
        if (
            len(args) == 1
            and isinstance(args[0], jnp.ndarray | np.ndarray)
            and args[0].shape == self.shape
        ):
            return args[0].nonzero()
        self = self._with_newaxes(args)
        if len(args) < self.ndim:
            args = args + (slice(None),) * (self.ndim - len(args))
        n_dummies = 0
        parsed = []
        for i, a in enumerate(args):
            axis = i - n_dummies
            n = self.shape[axis]
            if isinstance(a, slice):
                a = slice(*a.indices(self.shape[axis]))
            elif jnp.isscalar(a):
                a = self._handle_scalar(a, n)
            elif a is not None and not jnp.isscalar(a):
                a = jnp.asarray(a)
                # Bounds checking does not work with JIT compilation
                # if jnp.issubdtype(a.dtype, jnp.integer):
                #     a = jnp.where(a < 0, a + n, a)
                #     if a.min() < -n or a.max() >= n:
                #         errmsg = f"index out of bounds for axis {axis} with size {n}"
                #         raise IndexError(errmsg)
                if jnp.issubdtype(a.dtype, jnp.bool_):
                    if len(a) != n:
                        errmsg = (
                            f"boolean index did not match shape indexed array in index "
                            f"{axis}: got {a.shape}, expected {(n,)}"
                        )
                        raise IndexError(errmsg)
                    a = a.nonzero()
                elif not np.issubdtype(a.dtype, np.integer):
                    errmsg = f"invalid index type '{a.dtype}'"
                    raise IndexError(errmsg)
            parsed.extend(a) if isinstance(a, tuple) else parsed.append(a)
        return tuple(parsed)

    def _handle_scalar(self, a: int, n: int) -> int:
        if not isinstance(a, int | jnp.integer | np.integer) and not (
            isinstance(a, jnp.ndarray | np.ndarray) and jnp.isscalar(a)
        ):
            errmsg = f"expected integer index, got '{type(a)}'"
            raise TypeError(errmsg)
        a = lax.cond(a < 0, lambda x: x + n, lambda x: x, a)
        # if a < 0 or a >= n:
        #     errmsg = f"index {a} is out of bounds for axis with size {n}"
        #     raise IndexError(errmsg)
        return a

    def _handle_ellipses(self, args: tuple[IndexArg, ...]) -> tuple[IndexArg, ...]:
        if args is Ellipsis:
            return self[(slice(None),) * self.ndim]
        if not isinstance(args, tuple):
            args = (args,)
        try:
            # Handle expressions with ellipsis
            ellpos = args.index(Ellipsis)
            if Ellipsis in args[ellpos + 1 :]:
                errmsg = "an index expression can only contain a single ellipsis"
                raise ValueError(errmsg)
            left = args[:ellpos]
            right = args[ellpos + 1 :]
            lleft = sum(a is not None for a in left)
            lright = sum(a is not None for a in right)
            mid = (slice(None),) * (self.ndim - lleft - lright)
            return self[left + mid + right]
        except ValueError:
            pass
        return args

    def _with_newaxes(self, args: tuple[IndexArg, ...]) -> Self:
        """Return a new ShapedIndexExpression with new axes added.

        Parameters
        ----------
        args
            Indexing arguments, where `None` indicates a new axis.

        Returns
        -------
        ShapedIndexExpression
            New ShapedIndexExpression with updated shape.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from grgg.utils.indexing import ShapedIndexExpression
        >>> shape = (6, 5, 5)
        >>> shape_exp = ShapedIndexExpression(shape)
        >>> args = shape_exp[None, :, 1:3, None]
        >>> shape_exp._with_newaxes(args).shape
        (1, 6, 5, 1, 5)
        """
        shape = [
            1,
        ] * (self.ndim + sum(1 for a in args if a is None))
        n_dummies = 0
        for i in range(len(shape)):
            if i < len(args) and args[i] is None:
                n_dummies += 1
            else:
                shape[i] = self.shape[i - n_dummies]
        shape_exp = ShapedIndexExpression(*shape)
        if len(args) > shape_exp.ndim:
            errmsg = f"too many indices for shape {self.shape}"
            raise IndexError(errmsg)
        return shape_exp

    def _is_advanced_indexing_contiguous(self, args: tuple[IndexArg, ...]) -> bool:
        """Check if advanced indexing in args is contiguous.

        Parameters
        ----------
        args
            Indexing arguments.

        Returns
        -------
        bool
            True if advanced indexing is contiguous, False otherwise.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> shape = (6, 5, 5, 4, 4)
        >>> shape_exp = ShapedIndexExpression(shape)
        >>> args = shape_exp[:, [0, 2], [1, 3], ..., None]
        >>> shape_exp._is_advanced_indexing_contiguous(args)
        True
        >>> args = shape_exp[[0, 1], :, [1, 2], :, 1]
        >>> shape_exp._is_advanced_indexing_contiguous(args)
        False
        >>> args = shape_exp[:4, [0, 2], None, [1, 3]]
        >>> shape_exp._is_advanced_indexing_contiguous(args)
        False
        >>> shape = (3, 4, 5)
        >>> shape_exp = ShapedIndexExpression(shape)
        >>> args = shape_exp[1, :2, [0, 1, 2], None, None]
        >>> shape_exp._is_advanced_indexing_contiguous(args)
        False
        >>> args = shape_exp[:, :2, [0, 1, 2], None, None]
        >>> shape_exp._is_advanced_indexing_contiguous(args)
        True
        """
        adv_pos = [
            i for i, a in enumerate(args) if a is not None and not isinstance(a, slice)
        ]
        return max(adv_pos) - min(adv_pos) + 1 == len(adv_pos) if adv_pos else False


class IndexableShape(Shaped):
    """Indexable shape of an array-like object.

    Attributes
    ----------
    shape
        Shape of the array-like object.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> X = jnp.arange(150).reshape(6,5,5)
    >>> shape = IndexableShape(X.shape)
    >>> X[0].shape == shape[0]
    True
    >>> X[0, 1].shape == shape[0, 1]
    True
    >>> X[0, 1, 3].shape == shape[0, 1, 3]
    True
    >>> X[:, 1:3].shape == shape[:, 1:3]
    True
    >>> X[1:3, 2:4].shape == shape[1:3, 2:4]
    True
    >>> X[None, 1:3, :, None].shape == shape[None, 1:3, :, None]
    True
    >>> X[:3, [0, 2, 4]].shape == shape[:3, [0, 2, 4]]
    True
    >>> X[[0,1], [3, 4], :2].shape == shape[[0,1], [3, 4], :2]
    True
    >>> X[[0,1], [0,1], [0,2]].shape == shape[[0,1], [0,1], [0,2]]
    True
    >>> X[[[0,1],[0,2]], None, [0,1], 1].shape == shape[[[0,1],[0,2]], None, [0,1], 1]
    True

    Full boolean indexing.
    >>> from grgg import RandomGenerator
    >>> rng = RandomGenerator(0)
    >>> mask = rng.randint(X.shape, 0, 2).astype(bool)
    >>> X[mask].shape == shape[mask]
    True

    Partial boolean masking.
    >>> mask = jnp.asarray([True]*3 + [False]*3)
    >>> X[mask,].shape == shape[mask,]
    True
    >>> X[..., mask[:5]].shape == shape[..., mask[:5]]
    True

    Mixed indexing.
    >>> X[1, mask[:5], None, 1].shape == shape[1, mask[:5], None, 1]
    True
    """

    index_expr: ShapedIndexExpression

    def __init__(self, shape: tuple[int, ...] | ShapedIndexExpression) -> None:
        if isinstance(shape, tuple):
            shape = ShapedIndexExpression(shape)
        self.index_expr = shape

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.index_expr.equals(other.index_expr)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array-like object."""
        return self.index_expr.shape

    def __getitem__(self, args: IndexArg | tuple[IndexArg, ...]) -> tuple[int, ...]:
        if not isinstance(args, tuple):
            args = (args,)
        if not args or len(args) == 1 and args[0] is Ellipsis:
            return self.shape
        if len(args) == 1 and isinstance(args[0], jnp.ndarray | np.ndarray):
            dtype = args[0].dtype
            if jnp.issubdtype(dtype, jnp.bool) or np.issubdtype(dtype, bool):
                return self[args[0].nonzero()]
        args = self.index_expr[args]
        shape = self.index_expr._with_newaxes(args).shape
        basic, advanced = self._split_into_basic_and_advanced(args)
        # Handle basic indexing
        shapes = tuple(
            (len(range(*b.indices(s))),)
            if isinstance(b, slice)
            else (1,)
            if b is None
            else ()
            for b, s in zip(basic, shape, strict=True)
        )
        # Handle advanced indexing
        axes = [i for i, a in enumerate(advanced) if a is not None]
        if axes:
            broadcasted = jnp.broadcast_shapes(*(advanced[i].shape for i in axes))
            advanced = list(advanced)
            advanced[axes[0]] = broadcasted
            for i in axes[1:]:
                advanced[i] = ()
        broadcasted_shapes = []
        if self.index_expr._is_advanced_indexing_contiguous(args):
            # Contiguous advanced indexing
            # Add advanced indices groups after the first advanced index
            for s, a in zip(shapes, advanced, strict=True):
                if a is None:
                    broadcasted_shapes.append(s)
                else:
                    broadcasted_shapes.append(a)
        else:
            # Non-contiguous advanced indexing
            # Add all advanced shapes first, then all basic shapes
            broadcasted_shapes.extend(a for a in advanced if a is not None)
            broadcasted_shapes.extend(
                s for a, s in zip(advanced, shapes, strict=True) if a is None
            )
        return sum(broadcasted_shapes, start=())

    def _split_into_basic_and_advanced(
        self, args: tuple[IndexArg, ...]
    ) -> tuple[tuple[IndexArg, ...], tuple[IndexArg, ...]]:
        basic = []
        advanced = []
        for a in args:
            if a is None or isinstance(a, slice) or jnp.isscalar(a):
                basic.append(a)
                advanced.append(None)
            else:
                advanced.append(a)
                basic.append(slice(None))
        return tuple(basic), tuple(advanced)


class CartesianCoordinates(Shaped):
    """Converter of Numpy-style indexing to Cartesian coordinates
    relative to an array of a specified shape.

    Attributes
    ----------
    shape
        The shape of the array.
    """

    index: IndexableShape

    def __init__(
        self, shape: int | tuple[int, ...] | IndexableShape, *more_sizes: int
    ) -> None:
        if isinstance(shape, tuple | IndexableShape):
            if more_sizes:
                errmsg = "cannot specify shape as both tuple and individual sizes"
                raise ValueError(errmsg)
            if isinstance(shape, tuple):
                shape = IndexableShape(shape)
        else:
            shape = (shape, *more_sizes)
            shape = IndexableShape(shape)
        self.index = shape

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.index.equals(other.index)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array."""
        return self.index.shape

    @property
    def index_expr(self) -> ShapedIndexExpression:
        """Shape expression of the array."""
        return self.index.index_expr

    @property
    def i_(self) -> ShapedIndexExpression:
        """:class:`ShapedIndexExpression` instance for the given shape."""
        return self.index_expr

    @property
    def s_(self) -> IndexableShape:
        """:class:`IndexableShape` instance for the given shape."""
        return self.index

    def __getitem__(
        self, args: IndexArg | tuple[IndexArg, ...]
    ) -> tuple[jnp.ndarray, ...]:
        args = self.index_expr[args]
        # Broadcast advanced indices
        basic = {}
        advanced = {}
        for i, a in enumerate(args):
            if a is None or isinstance(a, slice) or jnp.isscalar(a):
                basic[i] = (
                    a if isinstance(a, slice) else jnp.asarray([0] if a is None else a)
                )
            else:
                advanced[i] = a
        adv_broadcasted = jnp.broadcast_arrays(*advanced.values()) if advanced else ()
        for k, a in zip(advanced, adv_broadcasted, strict=True):
            advanced[k] = a
        # Create a meshgrid of all non-scalar basic indices
        non_scalars = {
            k: jnp.r_[basic[k]] if isinstance(basic[k], slice) else basic[k]
            for k in basic
            if not jnp.isscalar(basic[k])
        }
        non_scalars_idx = jnp.ix_(*(jnp.r_[s] for s in non_scalars.values()))
        for k, a in zip(non_scalars, non_scalars_idx, strict=True):
            non_scalars[k] = a
            basic[k] = a
        # Determine proper axes ordering based on the resulting
        # position of the grouped advanced indices
        adv_pos = (
            0
            if not advanced
            else (
                min(advanced)
                if self.index_expr._is_advanced_indexing_contiguous(args)
                else 0
            )
        )
        if advanced:
            basic_ndim = non_scalars_idx[0].ndim if non_scalars_idx else 0
            adv_ndim = adv_broadcasted[0].ndim if adv_broadcasted else 0
            # Expand basic indices to accommodate advanced indices
            for k in basic:
                index = basic[k]
                if jnp.isscalar(index):
                    continue
                left = (slice(None),) * adv_pos
                expand = (None,) * adv_ndim
                basic[k] = index[*left, *expand, ...]
            # Expand advanced indices to accommodate basic indices
            for k in advanced:
                left = (None,) * adv_pos
                right = (None,) * (basic_ndim - adv_pos)
                advanced[k] = advanced[k][*left, ..., *right]
        joint_mapping = {**basic, **advanced}
        arrays = tuple(joint_mapping[i] for i in range(len(args)))
        return arrays


class MultiIndexRavel(Shaped):
    """Ravel arbitrary indexing into an array of flat indices.

    Attributes
    ----------
    shape
        Shape of the array to unravel indices into.
    mode
        Specifies how out-of-bounds indices are handled. Can be one of
        {'raise', 'wrap', 'clip'}. Default is 'clip'.
        Mode 'raise' is not compatible with JIT compilation.
    order
        Determines whether the multi-index is considered in 'C' (row-major)
        or 'F' (column-major) order. Default is 'C'.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> X = jnp.arange(100).reshape(10, 10)
    >>> ravel = MultiIndexRavel(X.shape, mode="raise")
    >>> def check(idx):
    ...     Y = X.flatten()[ravel[idx]]
    ...     return jnp.array_equal(Y, X[idx]).item()

    Test basic indexing.
    >>> idx = ravel.i_[...]
    >>> check(idx)
    True
    >>> idx = ravel.i_[1, 4]
    >>> check(idx)
    True
    >>> idx = ravel.i_[:5, 1:4]
    >>> check(idx)
    True
    >>> idx = ravel.i_[None, :5, 1:4, None]
    >>> check(idx)
    True

    Test advanced indexing.
    >>> idx = ravel.i_[[0, 2, 4], [1, 3, 4]]
    >>> check(idx)
    True
    >>> idx = ravel.i_[jnp.ix_(jnp.array([0, 2, 4]), jnp.array([1, 3, 4]))]
    >>> check(idx)
    True
    >>> idx = ravel.i_[jnp.arange(100).reshape(X.shape) % 2 == 1]
    >>> check(idx)
    True
    >>> mask = jnp.asarray([True]*4 + [False]*6)
    >>> idx = ravel.i_[mask, :]
    >>> check(idx)
    True
    """

    index: IndexableShape
    mode: str = eqx.field(static=True)
    order: str = eqx.field(static=True)

    def __init__(
        self,
        index: tuple[int, ...] | IndexableShape,
        mode: str = "clip",
        order: str = "C",
    ) -> None:
        if isinstance(index, tuple):
            index = IndexableShape(index)
        self.index = index
        self.mode = mode
        self.order = order

    def _equals(self, other: object) -> bool:
        return (
            super()._equals(other)
            and self.index.equals(other.index)
            and self.mode == other.mode
            and self.order == other.order
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array to unravel indices into."""
        return self.index.shape

    @property
    def i_(self) -> ShapedIndexExpression:
        """ShapedIndexExpression of the array to unravel indices into."""
        return self.index.index_expr

    @property
    def s_(self) -> IndexableShape:
        """IndexableShape of the array to unravel indices into."""
        return self.index

    @property
    def index_expr(self) -> ShapedIndexExpression:
        """Shape expression of the array to unravel indices into."""
        return self.index.index_expr

    @property
    def coords(self) -> CartesianCoordinates:
        """CartesianCoordinates of the array to unravel indices into."""
        return CartesianCoordinates(self.index)

    def __getitem__(
        self, args: IndexArg | tuple[IndexArg, ...]
    ) -> tuple[jnp.ndarray, ...]:
        if not isinstance(args, tuple):
            args = (args,)
        if not args or args == (Ellipsis,):
            return self._ravel(jnp.indices(self.shape, sparse=True))
        args = self.index_expr[args]
        coords = self.coords[args]
        indexed_shape = self.index_expr._with_newaxes(args).shape
        return self._ravel(coords, indexed_shape)

    def _ravel(
        self, coords: Sequence[jnp.ndarray], shape: tuple[int, ...] | None = None
    ) -> jnp.ndarray:
        """Ravel multi-dimensional coordinate indices into flat indices.

        Examples
        --------
        Here are some more examples (and tests) for raveling 3D indices and arrays.
        >>> import jax.numpy as jnp
        >>>
        >>> X = jnp.arange(60).reshape(3,4,5)
        >>> ravel = MultiIndexRavel(X.shape, mode="raise")
        >>> Y = X.flatten()[ravel[...]]
        >>> jnp.array_equal(Y, X).item()
        True
        >>>
        >>> # Test check
        >>> def check(idx):
        ...     Y = X.flatten()[ravel[idx]]
        ...     return jnp.array_equal(Y, X[idx]).item()
        >>>
        >>> idx = ravel.i_[...]
        >>> check(idx)
        True
        >>> idx = ravel.i_[1, 2, 3]
        >>> check(idx)
        True
        >>> idx = ravel.i_[:2, 1:3, 2:4]
        >>> check(idx)
        True
        >>> idx = ravel.i_[None, :2, 1:3, None, 2:4, None]
        >>> check(idx)
        True
        >>> idx = ravel.i_[[0, 2], [1, 3], [2, 4]]
        >>> check(idx)
        True
        >>> arrays = jnp.array([0, 2]), jnp.array([1, 3]), jnp.array([2, 4])
        >>> idx = ravel.i_[jnp.ix_(*arrays)]
        >>> check(idx)
        True
        >>> idx = ravel.i_[X % 2 == 1]
        >>> check(idx)
        True
        >>> mask = jnp.asarray([True]*2 + [False]*1)
        >>> idx = ravel.i_[mask, :, :]
        >>> check(idx)
        True
        >>> mask = jnp.array([True]*2 + [False]*2)
        >>> idx = ravel.i_[:, mask, :]
        >>> check(idx)
        True
        >>> mask = jnp.array([True]*3 + [False]*2)
        >>> idx = ravel.i_[:, :, mask]
        >>> check(idx)
        True
        >>> idx = ravel.i_[1, :2, mask]
        >>> check(idx)
        True
        >>> idx = ravel.i_[1, :2, mask, None, None]
        >>> check(idx)
        True
        >>> idx = ravel.i_[:, [0, 2], None, 1]
        >>> check(idx)
        True
        """
        if shape is None:
            shape = self.shape
        return jnp.ravel_multi_index(coords, shape, mode=self.mode, order=self.order)
