from jaxtyping import Array, Float, Integer

__all__ = ("Floats", "Scalar", "Vector", "Matrix", "Index", "Integers")

Floats = Float[Array, "..."]
Scalar = Float[Array, ""]  # Scalar value
Vector = Float[Array, "#values"]  # 1D array of node values
Matrix = Float[Array, "#rows #cols"]  # 2D array of node values

Index = Integer[Array, "#indices"]  # 1D array of integer indices
Integers = Integer[Array, "..."]  # Array of integer values
