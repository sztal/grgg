from beartype.vale import IsAttr, IsEqual
from jaxtyping import Array, Bool, Float, Integer, Num

__all__ = (
    "Reals",
    "Real",
    "RealVector",
    "RealMatrix",
    "Integers",
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

Reals = Float[Array, "..."]
Real = Float[Array, ""]  # Scalar value
RealVector = Float[Array, "#values"]  # 1D array of node values
RealMatrix = Float[Array, "#rows #cols"]  # 2D array of node values

Integers = Integer[Array, "..."]  # Array of integer values
Int = Integer[Array, ""]  # Scalar integer value
IntVector = Integer[Array, "#values"]  # 1D array of integer values
IntMatrix = Integer[Array, "#rows #cols"]  # 2D array of

Booleans = Bool[Array, "..."]  # Array of boolean values
Boolean = Bool[Array, ""]  # Scalar boolean value
BoolVector = Bool[Array, "#values"]  # 1D array of boolean values
BoolMatrix = Bool[Array, "#rows #cols"]  # 2D array of boolean values

IsHomogeneous = IsAttr["is_homogeneous", IsEqual[True]]
IsHeterogeneous = IsAttr["is_heterogeneous", IsEqual[True]]
