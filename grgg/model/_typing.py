from jaxtyping import Array, Float, Integer

__all__ = ("Floats", "Scalar", "Vector", "Matrix", "Integers", "IntVector", "IntMatrix")

Floats = Float[Array, "..."]
Scalar = Float[Array, ""]  # Scalar value
Vector = Float[Array, "#values"]  # 1D array of node values
Matrix = Float[Array, "#rows #cols"]  # 2D array of node values

Integers = Integer[Array, "..."]  # Array of integer values
IntVector = Integer[Array, "#values"]  # 1D array of integer values
IntMatrix = Integer[Array, "#rows #cols"]  # 2D array of
