import warnings
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg import options
from grgg.models.geometric.grgg.parameters import ParameterGroups
from grgg.utils.misc import split_kwargs_by_signature
from grgg.utils.quantize import ArrayQuantizer

from .model import GRGG


class QuantizedGRGG(GRGG):
    """A GRGG model with quantized parameters.

    For other attributes and details, see :class:`grgg.model.grgg.grgg.GRGG`.

    Attributes
    ----------
    quantizer
        Quantizer instances for handling (de)quantization of model parameters
        and computed quantities.
    spec
        Quantization specification for each layer in the model.
    """

    quantizer: ArrayQuantizer | None
    _quantized_names: list[tuple[str, ...]] = eqx.field(static=True, repr=False)

    def __init__(
        self,
        *args: Any,
        quantizer: ArrayQuantizer,
        _quantized_names: list[tuple[str, ...]],
        **kwargs: Any,
    ) -> None:
        """Initialization method.

        In general, users should not call this directly, but rather use
        :meth:`QuantizedGRGG.from_model`.

        Parameters
        ----------
        *args, **kwargs
            Pssed to the base class.
        quantizer
            Quantizer instance for handling (de)quantization of model parameters
            and computed quantities.
        _quantized_names
            Names of the heterogeneous parameters in the model
            (only heterogeneous parameters are quantized).
        _model_getter
            Function that returns the original non-quantized model.
        """
        cn = self.__class__.__name__
        msg = (
            f"direct initialization of '{cn}' detected, this is usually an error; "
            f"Use '{cn}.from_model()' constructor instead"
        )
        warnings.warn(msg, stacklevel=2, category=UserWarning)
        self.quantizer = quantizer
        self._quantized_names = _quantized_names
        super().__init__(*args, **kwargs)

    @property
    def is_quantized(self) -> bool:
        """Check if the model is quantized."""
        return True

    @property
    def n_codes(self) -> int:
        """Number of quantization codes."""
        return self.quantizer.n_codes

    @property
    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_codes

    @property
    def parameters(self) -> ParameterGroups:
        return super().parameters.replace(weights=self.quantizer.counts)

    def _equals(self, other: object) -> bool:
        if isinstance(other, GRGG):
            return other.equals(self)
        return (
            super()._equals(other)
            and self.quantizer.equals(other.quantizer)
            and self._quantized_names == other._quantized_names
        )

    @classmethod
    def from_model(cls, model: GRGG, n_codes: int | None = None, **kwargs: Any) -> Self:
        """Create a quantized model from a non-quantized model.

        Parameters
        ----------
        model
            Model to be quantized.
        n_codes
            Number of quantization codes to use.
        **kwargs
            Additional keyword arguments passed to the quantizer.
        """
        if not model.is_heterogeneous:
            errmsg = "only heterogeneous models can be quantized"
            raise ValueError(errmsg)
        if model.is_quantized:
            errmsg = "model is already quantized"
            raise ValueError(errmsg)
        if n_codes is None:
            n_codes = options.quantize.n_codes
        if n_codes >= model.n_nodes:
            warnings.warn(
                "trying to quantize with 'n_codes >= n_nodes', skipping quantization",
                category=UserWarning,
                stacklevel=2,
            )
            return model
        params = model.parameters.heterogeneous.array
        quantizer = ArrayQuantizer.from_data(params, n_codes=n_codes, **kwargs)
        quantized_params = quantizer.quantize(params)
        quantized_names = model.parameters.heterogeneous.names
        layers = []
        count = 0
        for layer, names in zip(model.layers, quantized_names, strict=True):
            new_params = {}
            for name in names:
                new_params[name] = quantized_params[:, count]
                count += 1
            layer = layer.detach().set_parameters(new_params)
            layers.append(layer)
        model_kwargs, _ = split_kwargs_by_signature(GRGG.__init__, **model.__dict__)
        model_kwargs.update(
            layers=layers,
            quantizer=quantizer,
            _quantized_names=quantized_names,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            return cls(**model_kwargs)

    def dequantize(self) -> GRGG:
        """Return the original non-quantized model."""
        params = self.quantizer.dequantize(self.parameters.heterogeneous.array)
        param_names = self._quantized_names
        layers = []
        count = 0
        for layer, names in zip(self.layers, param_names, strict=True):
            new_params = {}
            for name in names:
                new_params[name] = params[:, count]
                count += 1
            layer = layer.detach().set_parameters(new_params)
            layers.append(layer)
        model_kwargs, _ = split_kwargs_by_signature(GRGG.__init__, **self.__dict__)
        model_kwargs.update(layers=layers)
        return GRGG(**model_kwargs)

    def dequantize_arrays(self, *arrays: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Dequantize arrays of quantized parameters.

        Parameters
        ----------
        *arrays
            Arrays to be dequantized. Each array should have leading dimension
            of length equal to `n_codes`. It is assumed that the order of the
            arrays along the first dimension matches the order of the quantized
            parameters in the model.
        """
        params = self.parameters.heterogeneous.array
        return self.quantizer.dequantize(params, *arrays)[1:]
