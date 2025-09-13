from jaxtyping import Array, Bool, Float, Integer

__all__ = (
    "Floats",
    "Scalar",
    "Vector",
    "Matrix",
    "Integers",
    "IntVector",
    "IntMatrix",
    "BoolVector",
)

Floats = Float[Array, "..."]
Scalar = Float[Array, ""]  # Scalar value
Vector = Float[Array, "#values"]  # 1D array of node values
Matrix = Float[Array, "#rows #cols"]  # 2D array of node values

Integers = Integer[Array, "..."]  # Array of integer values
IntVector = Integer[Array, "#values"]  # 1D array of integer values
IntMatrix = Integer[Array, "#rows #cols"]  # 2D array of

Booleans = Bool[Array, "..."]  # Array of boolean values
BoolVector = Bool[Array, "#values"]  # 1D array of boolean values
