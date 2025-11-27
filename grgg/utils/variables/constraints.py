from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from grgg._options import options
from grgg._typing import Number
from grgg.utils.dispatch import dispatch
from grgg.utils.misc import logexp1m, logexp1p

__all__ = ("Constraints",)


def _to_scalar(x: ArrayLike) -> Number:
    """Convert a single-element array to a scalar."""
    if isinstance(x, jnp.ndarray):
        if x.ndim == 0:
            return x.item()
        errmsg = "expected a scalar array"
        raise ValueError(errmsg)
    return x


class Constraints(eqx.Module):
    """Abstract base class for variable constraints.

    Provides functionality for constraining variables to specific domains and bounds,
    with transformations between constrained and unconstrained spaces implemented
    for the real domain. This is useful for optimization where constraints need to be
    enforced while working in unconstrained parameter spaces.

    The class implements a dispatch-based transformation system that handles different
    combinations of bounds (lower only, upper only, both, or neither) through
    specialized methods. Transformations map constrained values to unconstrained
    spaces suitable for optimization, and inverse transformations recover the
    original constrained values.

    Attributes
    ----------
    domain
        Domain of the variable. One of "bool", "int", "real", or "complex".
    lower
        Lower bound for the variable. Default is None (unbounded from below).
    upper
        Upper bound for the variable. Default is None (unbounded from above).
    method
        Transformation method. Either "log" (standard log transform) or
        "logexp1m" (numerically stable log(exp(x) - 1) transform).
        Default is "logexp1m". Note that the latter is often better behaved when
        reparametrizing variables with thin tails.

    Notes
    -----
    The transformation logic works as follows:
    - No bounds: identity transformation (x -> x)
    - Lower bound only: shift and log transform (x -> log(x - lower))
    - Upper bound only: reflect, shift, and log transform (x -> log(upper - x))
    - Both bounds: logit-style transform mapping [lower, upper] to (-∞, ∞)

    Examples
    --------
    Create a constraint for real-valued variables bounded in [0, 1].

    >>> import jax.numpy as jnp
    >>> from grgg.utils.variables.constraints import Constraints
    >>> constraint = Constraints(domain="real", lower=0.0, upper=1.0)
    >>> x = jnp.array([0.3, 0.5, 0.8])

    Check if values satisfy the constraint (validates domain and bounds).

    >>> constraint.check(x)

    Transform to unconstrained space and back.

    >>> y = constraint.transform(x)
    >>> y.shape
    (3,)
    >>> x_recovered = constraint.inverse_transform(y)
    >>> jnp.allclose(x, x_recovered, atol=1e-6)
    Array(True, dtype=bool)

    Constraints with only a lower bound (positive reals).

    >>> pos_constraint = Constraints(domain="real", lower=0.0, upper=None)
    >>> x_pos = jnp.array([1.0, 2.5, 10.0])
    >>> pos_constraint.check(x_pos)
    >>> y_pos = pos_constraint.transform(x_pos)
    >>> x_pos_recovered = pos_constraint.inverse_transform(y_pos)
    >>> jnp.allclose(x_pos, x_pos_recovered, atol=1e-6)
    Array(True, dtype=bool)

    Constraints with only an upper bound.

    >>> upper_constraint = Constraints(domain="real", lower=None, upper=10.0)
    >>> x_upper = jnp.array([1.0, 5.0, 9.0])
    >>> upper_constraint.check(x_upper)
    >>> y_upper = upper_constraint.transform(x_upper)
    >>> x_upper_recovered = upper_constraint.inverse_transform(y_upper)
    >>> jnp.allclose(x_upper, x_upper_recovered, atol=1e-6)
    Array(True, dtype=bool)

    Unconstrained real values (identity transform, bounds default to None).

    >>> unconstrained = Constraints(domain="real")
    >>> x_free = jnp.array([-5.0, 0.0, 5.0])
    >>> unconstrained.check(x_free)
    >>> y_free = unconstrained.transform(x_free)
    >>> jnp.array_equal(x_free, y_free)
    Array(True, dtype=bool)

    Using different transformation methods.

    >>> constraint_log = Constraints(domain="real", lower=0.0, upper=None, method="log")
    >>> x_log = jnp.array([1.0, 2.0, 3.0])
    >>> y_log = constraint_log.transform(x_log)
    >>> jnp.allclose(y_log, jnp.log(x_log))
    Array(True, dtype=bool)

    Integer domain constraint (bounds optional, default to None).

    >>> int_constraint = Constraints(domain="int")
    >>> x_int = jnp.array([1, 2, 3], dtype=jnp.int32)
    >>> int_constraint.check(x_int)

    Bounds checking with valid and invalid values.

    >>> constraint = Constraints(domain="real", lower=0.0, upper=1.0)
    >>> x_valid = jnp.array([0.3, 0.5, 0.8])
    >>> constraint.check_bounds(x_valid)
    >>> x_invalid = jnp.array([-0.1, 0.5, 1.2])
    >>> try:
    ...     constraint.check_bounds(x_invalid)
    ... except ValueError as e:
    ...     "lower bound" in str(e) or "upper bound" in str(e)
    True
    """

    domain: Literal["bool", "int", "real", "complex"] = eqx.field(static=True)
    lower: Number | None = eqx.field(static=True, default=None, converter=_to_scalar)
    upper: Number | None = eqx.field(static=True, default=None, converter=_to_scalar)
    method: Literal["log", "logexp1m"] = eqx.field(
        static=True, default_factory=lambda: options.constraints.method
    )

    def check(self, x: ArrayLike) -> None:
        """Check if the variable satisfies the constraint.

        Parameters
        ----------
        x
            Variable to check.

        Raises
        ------
        ValueError
            If the variable does not satisfy the constraint.
        """
        self.check_domain(x, self.domain)

    def __check_init__(self) -> None:
        if self.domain == "complex" and self.has_bounds:
            errmsg = "bounds are not supported for complex variables"
            raise NotImplementedError(errmsg)

    @property
    def has_bounds(self) -> bool:
        """Whether the constraint has both lower and upper bounds."""
        return self.lower is not None or self.upper is not None

    @property
    def is_real(self) -> bool:
        """Whether the constraint domain is real."""
        return self.domain == "real"

    @dispatch.abstract
    def check_domain(
        self, x: ArrayLike, domain: Literal["bool", "int", "real", "complex"]
    ) -> None:
        """Check if the variable's domain matches the constraint's domain.

        Parameters
        ----------
        x
            Variable to check.
        domain
            Domain of the constraint.

        Raises
        ------
        ValueError
            If the variable's domain does not match the constraint's domain.
        """

    @check_domain.dispatch
    def _(self, x: ArrayLike, domain: Literal["bool"]) -> None:  # noqa
        if not jnp.issubdtype(x.dtype, jnp.bool_):
            errmsg = f"Expected boolean type, got {x.dtype}."
            raise ValueError(errmsg)

    @check_domain.dispatch
    def _(self, x: ArrayLike, domain: Literal["int"]) -> None:  # noqa
        if not jnp.issubdtype(x.dtype, jnp.integer):
            errmsg = f"Expected integer type, got {x.dtype}."
            raise ValueError(errmsg)

    @check_domain.dispatch
    def _(self, x: ArrayLike, domain: Literal["real"]) -> None:  # noqa
        if not jnp.isrealobj(x):
            errmsg = f"Expected real type, got {x.dtype}."
            raise ValueError(errmsg)

    @check_domain.dispatch
    def _(self, x: ArrayLike, domain: Literal["complex"]) -> None:  # noqa
        if not jnp.iscomplexobj(x):
            errmsg = f"Expected complex type, got {x.dtype}."
            raise ValueError(errmsg)

    def check_bounds(self, x: ArrayLike) -> None:
        """Check if the variable satisfies the bounds.

        Parameters
        ----------
        x
            Variable to check.

        Raises
        ------
        ValueError
            If the variable does not satisfy the bounds.
        """
        if self.lower is not None and jnp.any(x < self.lower):
            errmsg = f"lower bound on {self.lower} is not satisfied by {x}"
            raise ValueError(errmsg)
        if self.upper is not None and jnp.any(x > self.upper):
            errmsg = f"upper bound on {self.upper} is not satisfied by {x}"
            raise ValueError(errmsg)

    def transform(self, x: ArrayLike) -> ArrayLike:
        """Transform the variable to an unconstrained space.

        Parameters
        ----------
        x
            Variable to transform.

        Returns
        -------
        ArrayLike
            Transformed variable.
        """
        if self.domain != "real":
            errmsg = "transform is only implemented for real variables"
            raise NotImplementedError(errmsg)
        return self._transform_impl(x, self.lower, self.upper)

    def inverse_transform(self, y: ArrayLike) -> ArrayLike:
        """Inverse transform from the unconstrained space to the constrained space.

        Parameters
        ----------
        y
            Variable in the unconstrained space.

        Returns
        -------
        ArrayLike
            Transformed variable in the constrained space.
        """
        if self.domain != "real":
            errmsg = "inverse transform is only implemented for real variables"
            raise NotImplementedError(errmsg)
        return self._inverse_transform_impl(y, self.lower, self.upper)

    # Implementations ----------------------------------------------------------------

    @dispatch.abstract
    def _transform_impl(
        self,
        x: ArrayLike,
        lower: Number | None,
        upper: Number | None,
    ) -> ArrayLike:
        """Transform the variable to an unconstrained space."""

    @_transform_impl.dispatch
    def _(self, x: ArrayLike, lower: None, upper: None) -> ArrayLike:  # noqa
        return x

    @_transform_impl.dispatch
    def _(self, x: ArrayLike, lower: Any, upper: None) -> ArrayLike:  # noqa
        shifted = x - lower
        return self._shifted_to_unconstrained(shifted)

    @_transform_impl.dispatch
    def _(self, x: ArrayLike, lower: None, upper: Any) -> ArrayLike:  # noqa
        shifted = upper - x
        return self._shifted_to_unconstrained(shifted)

    @_transform_impl.dispatch
    def _(self, x: ArrayLike, lower: Any, upper: Any) -> ArrayLike:  # noqa
        p = (x - lower) / (upper - lower)
        odds = p / (1 - p)
        return self._shifted_to_unconstrained(odds)

    @dispatch.abstract
    def _inverse_transform_impl(
        self,
        y: ArrayLike,
        lower: Number | None,
        upper: Number | None,
    ) -> ArrayLike:
        """Inverse transform from the unconstrained space to the constrained space."""

    @_inverse_transform_impl.dispatch
    def _(self, y: ArrayLike, lower: None, upper: None) -> ArrayLike:  # noqa
        return y

    @_inverse_transform_impl.dispatch
    def _(self, y: ArrayLike, lower: Any, upper: None) -> ArrayLike:  # noqa
        shifted = self._unconstrained_to_shifted(y)
        return shifted + lower

    @_inverse_transform_impl.dispatch
    def _(self, y: ArrayLike, lower: None, upper: Any) -> ArrayLike:  # noqa
        shifted = self._unconstrained_to_shifted(y)
        return upper - shifted

    @_inverse_transform_impl.dispatch
    def _(self, y: ArrayLike, lower: Any, upper: Any) -> ArrayLike:  # noqa
        shifted = self._unconstrained_to_shifted(y)
        p = shifted / (1 + shifted)
        return lower + p * (upper - lower)

    def _shifted_to_unconstrained(self, x: ArrayLike) -> ArrayLike:
        return jnp.log(x) if self.method == "log" else logexp1m(x)

    def _unconstrained_to_shifted(self, y: ArrayLike) -> ArrayLike:
        return jnp.exp(y) if self.method == "log" else logexp1p(y)
