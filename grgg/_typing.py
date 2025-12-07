import jax.numpy as jnp
from beartype.vale import Is, IsAttr, IsEqual
from jaxtyping import Array, Bool, Integer, Num
from jaxtyping import Float as _Float

__all__ = (
    "Reals",
    "Real",
    "RealVector",
    "RealMatrix",
    "Integers",
    "Floats",
    "Float",
    "FloatVector",
    "FloatMatrix",
    "Int",
    "IntVector",
    "IntMatrix",
    "Booleans",
    "Boolean",
    "BoolVector",
    "BoolMatrix",
    "Numbers",
    "Number",
    "Vector",
    "Matrix",
    "IsHomogeneous",
    "IsHeterogeneous",
)

Numbers = Num[Array, "..."]  # Array of numeric values (float or int)
Number = Num[Array, ""]  # Scalar numeric value (float or int)
Vector = Num[Array, "#values"]  # 1D array of numeric values (float or int)
Matrix = Num[Array, "#rows #cols"]  # 2D array of numeric values (float or int)

Floats = _Float[Array, "..."]
Float = _Float[Array, ""]  # Scalar value
FloatVector = _Float[Array, "#values"]  # 1D array of node values
FloatMatrix = _Float[Array, "#rows #cols"]  # 2D array of node values

Integers = Integer[Array, "..."]  # Array of integer values
Int = Integer[Array, ""]  # Scalar integer value
IntVector = Integer[Array, "#values"]  # 1D array of integer values
IntMatrix = Integer[Array, "#rows #cols"]  # 2D array of

Reals = Floats | Integers  # Array of real values
Real = Float | Int  # Scalar real value
RealVector = FloatVector | IntVector  # 1D array of real values
RealMatrix = FloatMatrix | IntMatrix  # 2D array of real values

Booleans = Bool[Array, "..."]  # Array of boolean values
Boolean = Bool[Array, ""]  # Scalar boolean value
BoolVector = Bool[Array, "#values"]  # 1D array of boolean values
BoolMatrix = Bool[Array, "#rows #cols"]  # 2D array of boolean values

IsScalar = Is[lambda x: jnp.asarray(x).shape == ()]
IsHomogeneous = IsAttr["is_homogeneous", IsEqual[True]]
IsHeterogeneous = IsAttr["is_heterogeneous", IsEqual[True]]
