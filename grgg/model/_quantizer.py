from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg.abc import AbstractModule
from grgg.quantize import ArrayQuantizer

if TYPE_CHECKING:
    from grgg.model.grgg import GRGG


SpecsT = tuple[Mapping[str, int], ...]


class ModelQuantizer(AbstractModule):
    """Model quantization handler."""

    quantizer: ArrayQuantizer
    specs: SpecsT = eqx.field(static=True)
    original: jnp.ndarray = eqx.field(repr=False)
    quantized: jnp.ndarray = eqx.field(repr=False)

    def __check_init__(self) -> None:
        if self.original.shape[1:] != self.quantized.shape[1:]:
            errmsg = "'original' and 'quantized' must have the same shape past axis 0"
            raise ValueError(errmsg)

    @property
    def n_codes(self) -> int:
        """Number of quantization codes."""
        return self.quantizer.n_codes

    @property
    def counts(self) -> jnp.ndarray:
        """Counts of each quantization code in the quantized parameters."""
        return self.quantizer.counts

    @classmethod
    def from_model(cls, model: "GRGG", **kwargs: Any) -> Self:
        """Create a quantizer from a model.

        Parameters
        ----------
        model
            Model to be quantized.
        **kwargs
            Additional keyword arguments passed to the quantizer.

        Returns
        -------
        ModelQuantizer
            Quantizer instance.
        """
        if not model.is_heterogeneous:
            errmsg = "model must be heterogeneous to be quantized"
            raise ValueError(errmsg)
        if model.is_quantized:
            errmsg = "model is already quantized"
            raise ValueError(errmsg)
        params, specs = cls._collect_parameters(model)
        quantizer = ArrayQuantizer.from_data(params, **kwargs)
        quantized = quantizer.quantize(params)
        return cls(
            quantizer=quantizer,
            specs=specs,
            original=params,
            quantized=quantized,
        )

    def equals(self, other: Any) -> bool:
        return (
            super().equals(other)
            and self.quantizer == other.quantizer
            and self.specs == other.specs
            and jnp.array_equal(self.original, other.original)
            and jnp.array_equal(self.quantized, other.quantized)
        )

    def quantize(self, model: "GRGG") -> "GRGG":
        """Return a quantized copy of the model."""
        return self._set_parameters(model, self.quantized, quantized=True)

    def dequantize(self, model: "GRGG") -> "GRGG":
        """Return a dequantized copy of the model."""
        if not model.is_quantized:
            return model
        return self._set_parameters(model, self.original, quantized=False)

    @classmethod
    def _collect_parameters(cls, model: "GRGG") -> tuple[jnp.ndarray, SpecsT]:
        params = []
        specs = []
        i = 0
        for layer in model.layers:
            lp = {k: v for k, v in layer.parameters.items() if not jnp.isscalar(v)}
            params.extend(lp.values())
            spec = {k: i + j for j, k in enumerate(lp.keys())}
            specs.append(spec)
            i += len(spec)
        params = jnp.column_stack(params)
        return params, tuple(specs)

    def _set_parameters(
        self,
        model: "GRGG",
        parameters: jnp.ndarray,
        *,
        quantized: bool = True,
    ) -> "GRGG":
        """Set the model parameters from a parameter array."""
        layers = []
        for i, spec in enumerate(self.specs):
            layer = model.layers[i]
            params = {k: parameters[:, j] for k, j in spec.items()}
            layer = layer.detach().set_parameters(**params)
            layers.append(layer)
        return model.replace(layers=layers, quantizer=self if quantized else None)
