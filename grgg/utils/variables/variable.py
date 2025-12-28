from collections.abc import Callable
from typing import Any, ClassVar, Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike, DTypeLike

from grgg._typing import Numbers
from grgg.utils.indexing import Shaped
from grgg.utils.misc import format_array

from .constraints import Constraints

__all__ = ("Variable",)


@format_array.dispatch
def _(x: "Variable") -> str:
    return f"{x.__class__.__name__}({format_array(x.data)})"


class Variable(Shaped):
    """Array variable with optional constraint-based transformations.

    A JAX-compatible array wrapper that supports optional constraint enforcement
    and automatic transformations between constrained and unconstrained spaces.
    When a real-valued constraint is attached and `transform=True`, the variable
    automatically transforms input data to an unconstrained space for optimization
    and inverse-transforms when accessed as a JAX array.

    Attributes
    ----------
    data
        Array data. If `is_transformable` is True (has a real constraint with
        `transform=True`), this stores the transformed unconstrained representation.
        Otherwise, it stores the original constrained values.
    transform
        Whether the variable should be internally represented in the transformed
        (unconstrained) space. Default is False. When True, enables automatic
        transformation to/from unconstrained space, which is useful for
        optimization with constrained variables. Only takes effect when a
        real-valued constraint with bounds is set.
    constraint
        Optional constraint specification (default is None). When set to a
        real-valued constraint with `transform=True`, the variable becomes
        transformable and automatically handles transformations.

    Notes
    -----
    The transformation behavior depends on the constraint and `transform` flag:
    - No constraint: data is stored and accessed directly
    - Non-real constraint: constraint is checked but no transformation applied
    - Real constraint with `transform=False`: constraint is checked only
    - Real constraint with `transform=True`: data is transformed to unconstrained
      space on input and inverse-transformed to constrained space on access
      via `__jax_array__`

    The `is_transformable` property returns True only when all conditions are met:
    `transform=True`, constraint is not None, and constraint domain is real.

    The class supports unary operations (`+`, `-`) and binary operations (`+`, `-`,
    `*`, `/`) that preserve the Variable type. Operations are performed on the
    underlying `data` attribute while maintaining the constraint and transform settings.

    Examples
    --------
    Create a simple unconstrained variable.

    >>> import jax.numpy as jnp
    >>> from grgg.utils.variables import Variable
    >>> x = Variable(jnp.array([1.0, 2.0, 3.0]))
    >>> x.data
    Array([1., 2., 3.], dtype=float32)
    >>> x.shape
    (3,)
    >>> x.dtype
    dtype('float32')

    Create a constrained variable without transformation (transform=False by default).

    >>> from grgg.utils.variables import Constraints
    >>> class BoundedVariable(Variable):
    ...     constraint = Constraints(domain="real", lower=0.0, upper=1.0)
    >>> y = BoundedVariable(jnp.array([0.3, 0.5, 0.8]))
    >>> y.is_transformable
    False
    >>> y.data
    Array([0.3, 0.5, 0.8], dtype=float32)

    Enable transformation for optimization by setting transform=True.

    >>> y_transformed = BoundedVariable(jnp.array([0.3, 0.5, 0.8]), transform=True)
    >>> y_transformed.is_transformable
    True

    The data is stored in transformed (unconstrained) space.

    >>> y_transformed.data.shape
    (3,)

    Accessing as JAX array returns the original constrained values.

    >>> y_array = jnp.asarray(y_transformed)
    >>> jnp.allclose(y_array, jnp.array([0.3, 0.5, 0.8]), atol=1e-6)
    Array(True, dtype=bool)

    Variable with a lower bound only (upper bound defaults to None).

    >>> class PositiveVariable(Variable):
    ...     constraint = Constraints(domain="real", lower=0.0)
    >>> z = PositiveVariable(jnp.array([1.0, 2.5, 10.0]), transform=True)
    >>> z.is_transformable
    True
    >>> z_array = jnp.asarray(z)
    >>> jnp.allclose(z_array, jnp.array([1.0, 2.5, 10.0]), atol=1e-6)
    Array(True, dtype=bool)

    Indexing operations.

    >>> x = Variable(jnp.array([1.0, 2.0, 3.0, 4.0]))
    >>> x_slice = x[:2]
    >>> x_slice.data
    Array([1., 2.], dtype=float32)

    Scalar variables.

    >>> scalar = Variable(jnp.array(5.0))
    >>> scalar.is_scalar
    True
    >>> scalar.shape
    ()

    Type conversion.

    >>> x_int = x.astype(jnp.int32)
    >>> x_int.dtype
    dtype('int32')

    Non-transformable constraint (integer domain, bounds default to None).

    >>> class IntVariable(Variable):
    ...     constraint = Constraints(domain="int")
    >>> int_var = IntVariable(jnp.array([1, 2, 3], dtype=jnp.int32))
    >>> int_var.is_transformable
    False
    >>> int_var.data
    Array([1, 2, 3], dtype=int32)

    Unary operations preserve the Variable type.

    >>> x = Variable(jnp.array([1.0, 2.0, 3.0]))
    >>> neg_x = -x
    >>> isinstance(neg_x, Variable)
    True
    >>> neg_x.data
    Array([-1., -2., -3.], dtype=float32)
    >>> pos_x = +x
    >>> jnp.array_equal(pos_x.data, x.data)
    Array(True, dtype=bool)

    Binary operations preserve the Variable type.

    >>> x = Variable(jnp.array([1.0, 2.0, 3.0]))
    >>> y = Variable(jnp.array([4.0, 5.0, 6.0]))
    >>> z = x + y
    >>> isinstance(z, Variable)
    True
    >>> z.data
    Array([5., 7., 9.], dtype=float32)

    Operations with scalars and arrays.

    >>> x2 = x * 2.0
    >>> x2.data
    Array([2., 4., 6.], dtype=float32)
    >>> x_scaled = x + jnp.array([10.0, 20.0, 30.0])
    >>> x_scaled.data
    Array([11., 22., 33.], dtype=float32)

    Operations preserve constraint and transform settings.

    >>> y_transformed = BoundedVariable(jnp.array([0.3, 0.5, 0.8]), transform=True)
    >>> y_doubled = y_transformed * 2.0
    >>> y_doubled.is_transformable
    True
    >>> isinstance(y_doubled, BoundedVariable)
    True
    """

    data: Numbers
    transform: bool = eqx.field(static=True)

    constraint: ClassVar[Constraints | None] = None

    def __init__(self, data: ArrayLike, *, transform: bool = False) -> None:
        self.transform = transform
        data = jnp.asarray(data)
        if self.is_transformable:
            data = self.constraint.transform(data)
        elif self.constraint is not None:
            self.constraint.check(data)
        self.data = data

    def __jax_array__(self) -> jnp.ndarray:
        if self.is_transformable:
            return self.constraint.inverse_transform(self.data)
        return self.data

    def __repr__(self) -> str:
        if not isinstance(self.data, jnp.ndarray):
            inner = self.data
        else:
            inner = self.data.item() if self.is_scalar else format_array(self.data)
        return f"{self.__class__.__name__}({inner})"

    def __getitem__(self, args: Any) -> jnp.ndarray:
        return self.replace(data=self.data[args])

    def __len__(self) -> int:
        return len(self.data)

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return self.replace(data=-self.data)

    def __add__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.add, other)

    def __radd__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.add, other, rev=True)

    def __sub__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.subtract, other)

    def __rsub__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.subtract, other, rev=True)

    def __mul__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.multiply, other)

    def __rmul__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.multiply, other, rev=True)

    def __truediv__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.divide, other)

    def __rtruediv__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.divide, other, rev=True)

    def __pow__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.power, other)

    def __rpow__(self, other: ArrayLike) -> Self:
        return self.__binop(jnp.power, other, rev=True)

    def __binop(
        self,
        op: Callable[[Self, ArrayLike], Self],
        other: ArrayLike,
        rev: bool = False,
    ) -> Self:
        if isinstance(other, Variable) and type(self) is not type(other):
            errmsg = f"cannot mix variable types: {type(self)} and {type(other)}"
            raise TypeError(errmsg)
        args = (self, jnp.asarray(other)) if not rev else (jnp.asarray(other), self)
        return self.replace(data=op(*args))

    @property
    def is_transformable(self) -> bool:
        """Whether the variable is transformable via its constraint."""
        return (
            self.transform and self.constraint is not None and self.constraint.is_real
        )

    @property
    def dtype(self) -> DTypeLike:
        """Data type of the array data."""
        return self.data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array data."""
        return self.data.shape

    @property
    def is_scalar(self) -> bool:
        """Whether the array is a scalar."""
        return self.data.ndim == 0

    def astype(self, dtype: DTypeLike) -> Self:
        return self.replace(data=self.data.astype(dtype))
