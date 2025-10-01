from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jax.lax import stop_gradient

from grgg._typing import IntVector, Real, RealVector
from grgg.abc.modules import AbstractModule
from grgg.utils.lazy import LazyOuter
from grgg.utils.misc import format_array

ParamT = Real | RealVector


class CONSTRAINTS(eqx.Enumeration):
    """Enumeration of supported parameter constraints."""

    real = "real"
    non_negative = "non-negative"


class AbstractParameterSpecification(ABC):
    """Abstract base class for parameter specifications."""

    def __call__(self, value: jnp.ndarray | None = None) -> jnp.ndarray:
        return self.validate(value)

    @property
    @abstractmethod
    def name(self) -> str:
        """Parameter name."""

    @property
    @abstractmethod
    def constraints(self) -> list[eqx._enum.EnumerationItem, ...]:
        """List of constraints."""
        return []

    @property
    @abstractmethod
    def default_value(self) -> ParamT:
        """Default parameter value."""

    @classmethod
    def validate(cls, value: jnp.ndarray | None) -> None:
        """Check all constraints."""
        spec = cls()
        if value is None:
            value = spec.default_value
        value = cls._validate(value)
        for constraint in spec.constraints:
            spec._check_constraint(value, constraint)
        return value

    @classmethod
    @abstractmethod
    def _validate(cls, value: jnp.ndarray) -> jnp.ndarray:
        """Validate input value."""
        value = jnp.asarray(value)
        if value.size <= 0:
            errmsg = "parameter value must be a non-empty array"
            raise ValueError(errmsg)
        return value

    def _check_constraint(
        self, value: jnp.ndarray, constraint: eqx._enum.EnumerationItem
    ) -> None:
        """Check constraint."""
        if constraint == CONSTRAINTS.real:
            self._error_if(constraint, ~jnp.isreal(value).all())
        elif constraint == CONSTRAINTS.non_negative:
            self._error_if(constraint, (value < 0).any())
        else:
            errmsg = f"unknown constraint: {constraint}"
            raise ValueError(errmsg)

    def _error_if(self, constraint, condition: bool) -> None:
        name = constraint._enumeration[constraint]
        if condition:
            errmsg = f"'{self.name}' must be {name}"
            raise ValueError(errmsg)


class AbstractNodeParameterSpecification(AbstractParameterSpecification):
    """Abstract base class for node parameter specifications."""

    @property
    def constraints(self) -> list[eqx._enum.EnumerationItem, ...]:
        return [*super().constraints, CONSTRAINTS.real]

    @classmethod
    def _validate(cls, value: jnp.ndarray) -> jnp.ndarray:
        value = super()._validate(value)
        if value.ndim > 1:
            errmsg = "node parameter value must be a scalar or a 1D array"
            raise ValueError(errmsg)
        return value

    @classmethod
    def validate(cls, value: jnp.ndarray) -> jnp.ndarray:
        """Check all constraints."""
        return super().validate(value).astype(float)


class Beta(AbstractNodeParameterSpecification):
    """Beta parameter (inverse temperature).

    It controls the strength of the coupling between the network topology
    and the underlying geometry.

    Attributes
    ----------
    value
        Parameter value. Default is 1.5.

    Examples
    --------
    >>> beta = Beta()
    >>> beta()  # default value
    Array(1.5, ...)
    >>> beta(2.0)  # homogeneous value
    Array(2.0, ...)
    >>> beta([1,2,3])  # heterogeneous value
    Array([1., 2., 3.], ...)

    Error is raised for invalid values.
    >>> beta(-1)  # negative value
    Traceback (most recent call last):
        ...
    ValueError: 'beta' must be non-negative
    """

    @property
    def name(self) -> str:
        return "beta"

    @property
    def constraints(self) -> list[eqx._enum.EnumerationItem, ...]:
        return [*super().constraints, CONSTRAINTS.non_negative]

    @property
    def default_value(self) -> ParamT:
        return jnp.array(1.5)


class Mu(AbstractNodeParameterSpecification):
    """Mu parameter (chemical potential).

    It controls the average degree of the network.

    Attributes
    ----------
    value
        Parameter value. Default is 0.0.

    Examples
    --------
    >>> mu = Mu()
    >>> mu()  # default value
    Array(0.0, ...)
    >>> mu(1.0)  # homogeneous value
    Array(1.0, ...)
    >>> mu([1, 2, 3])  # heterogeneous value
    Array([1., 2., 3.], ...)

    Error is raised for invalid values.
    >>> mu(1+1j)  # negative value
    Traceback (most recent call last):
        ...
    ValueError: 'mu' must be real
    """

    @property
    def name(self) -> str:
        return "mu"

    @property
    def default_value(self) -> ParamT:
        return jnp.array(0.0)


class AbstractParametersContainer(AbstractModule):
    """Abstract base class for parameter containers."""

    @property
    @abstractmethod
    def lengths(self) -> IntVector:
        """Lengths of parameter arrays in each layer."""

    @property
    def n_units(self) -> int:
        """Number of units (axis 0 length)."""
        return int(max(self.lengths))

    @property
    @abstractmethod
    def is_heterogeneous(self) -> bool:
        """Check if any parameter is heterogeneous."""

    @property
    @abstractmethod
    def heterogeneous(self) -> Self:
        """Select only heterogeneous parameters."""

    @property
    def aligned(self) -> Self:
        """Return parameters aligned to the number of units."""

    @property
    @abstractmethod
    def array(self) -> jnp.ndarray:
        """Parameters as a single array obtained with :func:`jax.numpy.column_stack`."""

    @property
    @abstractmethod
    def names(self) -> list[str] | list[tuple[str, ...]]:
        """Names of parameters in the container."""

    @abstractmethod
    def validate(self, n: int) -> Self:
        """Validate that parameter arrays have correct lengths."""

    @abstractmethod
    def dump(self) -> dict[str, list[float] | float]:
        """Dump parameters to a simple data structure."""


class Parameters(AbstractParametersContainer, Mapping[str, jnp.ndarray]):
    """Layer parameter mapping."""

    class Outer(eqx.Module, Mapping[str, LazyOuter]):
        """Lazy outer sums of parameter vectors."""

        parameters: "Parameters"

        def __getitem__(self, key: str) -> LazyOuter:
            param = self.parameters[key]
            if jnp.isscalar(param):
                return LazyOuter(param / 2, op=jnp.add)
            return LazyOuter(param, op=jnp.add)

        def __getattr__(self, name: str) -> LazyOuter:
            try:
                return self[name]
            except KeyError as err:
                errmsg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
                raise AttributeError(errmsg) from err

        def __iter__(self) -> Iterator[str]:
            yield from self.parameters

        def __len__(self) -> int:
            return len(self.parameters)

    _mapping: Mapping[str, jnp.ndarray]

    def __init__(
        self,
        _mapping: Mapping[str, jnp.ndarray] | None = None,
        **kwargs: jnp.ndarray,
    ) -> None:
        if _mapping is None:
            _mapping = {}
        _mapping = {**_mapping, **kwargs}
        self._mapping = {n: jnp.asarray(p) for n, p in _mapping.items()}

    def __repr__(self) -> str:
        items = ", ".join(f"{n}={format_array(p)}" for n, p in self._mapping.items())
        return f"{self.__class__.__name__}({items})"

    def __getitem__(self, key: str) -> jnp.ndarray:
        return self._mapping[key]

    def __getattr__(self, name: str) -> jnp.ndarray:
        try:
            return self._mapping[name]
        except KeyError as err:
            errmsg = f"'{self.__class__.__name__}' object has no attribute '{name}'"
            raise AttributeError(errmsg) from err

    def __iter__(self) -> Iterator[str]:
        yield from self._mapping

    def __len__(self) -> int:
        return len(self._mapping)

    def __check_init__(self) -> None:
        lengths = self.lengths
        n_units = max(lengths or [1])
        if not all(length in (1, n_units) for length in lengths):
            errmsg = "all parameter arrays must have the same length or be scalars"
            raise ValueError(errmsg)

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and len(self) == len(other)
            and all(
                k1 == k2 and jnp.array_equal(v1, v2)
                for (k1, v1), (k2, v2) in zip(self.items(), other.items(), strict=True)
            )
        )

    @property
    def outer(self) -> "Parameters.Outer":
        """Lazy outer sums of parameter vectors."""
        return self.Outer(self)

    @property
    def lengths(self) -> list[int]:
        """Lengths of parameter arrays in the layer."""
        return [1 if jnp.isscalar(p) else len(p) for p in self._mapping.values()]

    @property
    def aligned(self) -> Self:
        """Return parameters aligned to the number of units."""
        if not self._mapping:
            return self
        values = [
            v / 2 if self.is_heterogeneous and jnp.isscalar(v) else v
            for v in self._mapping.values()
        ]
        arrays = jnp.broadcast_arrays(*values)
        mapping = dict(zip(self._mapping.keys(), arrays, strict=True))
        return replace(self, _mapping=mapping)

    @property
    def is_heterogeneous(self) -> bool:
        """Check if any parameter is heterogeneous."""
        return any(not jnp.isscalar(p) for p in self._mapping.values())

    @property
    def heterogeneous(self) -> Self:
        """Select only heterogeneous parameters."""
        return replace(
            self,
            _mapping={n: p for n, p in self._mapping.items() if not jnp.isscalar(p)},
        )

    @property
    def array(self) -> jnp.ndarray:
        """Parameters as a single array."""
        if not self._mapping:
            return jnp.array([])
        return jnp.column_stack(list(self.aligned.values()))

    @property
    def names(self) -> tuple[str]:
        """Names of parameters in the container."""
        return tuple(self._mapping.keys())

    def validate(self, n: int) -> Self:
        """Validate that parameter arrays have correct lengths."""
        if not all(length in (1, n) for length in self.lengths):
            errmsg = "all parameter arrays must have the same length or be scalars"
            raise ValueError(errmsg)
        return self

    def dump(self) -> dict[str, jnp.ndarray]:
        """Dump parameters to a dictionary."""
        return self._mapping.copy()


class ParameterGroups(AbstractModule, Sequence[Parameters]):
    """Container for model parameters.

    Attributes
    ----------
    groups
        Tuple of parameter mappings.
    """

    _groups: tuple[Parameters, ...]
    weights: jnp.ndarray = eqx.field(converter=jnp.asarray)

    def __init__(
        self,
        *groups: Parameters,
        weights: jnp.ndarray = 1.0,
        _groups: Iterable[Parameters] = (),
    ) -> None:
        params = []
        groups = (*_groups, *groups)
        for group in groups:
            if isinstance(group, Mapping):
                params.append(group)
            else:
                params.extend(group)
        self._groups = tuple(Parameters(**param) for param in params)
        self.weights = stop_gradient(jnp.asarray(weights).astype(float))

    def __repr__(self) -> str:
        groups = "\n".join(f"    {repr(group)}" for group in self._groups)
        groups += "\n    weights=" + eqx.tree_pformat(self.weights)
        return f"{self.__class__.__name__}(\n{groups}\n)"

    def __check_init__(self) -> None:
        lengths = {*sum((g.lengths for g in self._groups), start=[])}
        if lengths and lengths - {1, max(lengths)}:
            errmsg = "all parameter arrays must have the same length or be scalars"
            raise ValueError(errmsg)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index: int) -> ParamT:
        return self._groups[index]

    @property
    def lengths(self) -> list[int]:
        """Lengths of parameter arrays in each layer."""
        return [group.lengths for group in self._groups]

    @property
    def n_units(self) -> int:
        """Number of units (nodes) in the network."""
        return max(length for group in self._groups for length in group.lengths)

    @property
    def aligned(self) -> Self:
        """Return parameters aligned to the number of units."""
        return replace(self, _groups=tuple(g.aligned for g in self._groups))

    @property
    def is_heterogeneous(self) -> bool:
        """Check if any parameter is heterogeneous."""
        return any(g.is_heterogeneous for g in self._groups)

    @property
    def heterogeneous(self) -> bool:
        """Select only heterogeneous parameters."""
        return replace(self, _groups=tuple(g.heterogeneous for g in self._groups))

    @property
    def array(self) -> jnp.ndarray:
        """Parameters as a single array."""
        arrays = []
        for group in self.aligned:
            arr = group.array
            if arr.size <= 0:
                continue
            arrays.append(arr)
        return jnp.column_stack(arrays) if arrays else jnp.array([])

    @property
    def names(self) -> list[tuple[str, ...]]:
        """Names of parameters in the container.

        Examples
        --------
        >>> from grgg import GRGG, Similarity, Complementarity
        >>> model = (
        ...     GRGG(2, 2) +
        ...     Similarity([0, 1], 1.0) +
        ...     Complementarity(2, [0, 1])
        ... )
        >>> model.parameters.names
        [('mu', 'beta'), ('mu', 'beta')]
        >>> model.parameters.heterogeneous.names
        [('mu',), ('beta',)]
        >>> model = GRGG(2, 2) + Similarity
        >>> model.parameters.names
        [('mu', 'beta')]
        >>> model.parameters.heterogeneous.names
        [()]
        """
        return [group.names for group in self._groups]

    def equals(self, other: object) -> bool:
        return (
            super().equals(other)
            and len(self) == len(other)
            and all(
                g1.equals(g2)
                for g1, g2 in zip(self._groups, other._groups, strict=True)
            )
            and jnp.array_equal(self.weights, other.weights)
        )

    def validate(self, n: int) -> Self:
        """Validate that parameter arrays have correct lengths."""
        for group in self:
            group.validate(n)
        return self

    def dump(self) -> list[dict[str, jnp.ndarray]]:
        """Dump parameters to a list of dictionaries."""
        return [group.dump() for group in self._groups]
