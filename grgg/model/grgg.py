from collections.abc import Callable, Iterable, Sequence
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg._typing import Floats
from grgg.manifolds import CompactManifold, Sphere

from .abc import AbstractModel
from .functions import (
    CouplingFunction,
    ProbabilityFunction,
)
from .layers import AbstractLayer

if TYPE_CHECKING:
    from ._sampling import Sample


class GRGG(AbstractModel, Sequence[Self]):
    """Generalized Random Geometric Graph model.

    Attributes
    ----------
    n_nodes
        Number of nodes in the model.
    manifold
        Compact isotropic manifold where nodes are embedded.
        Currently supported manifolds are:
        - :class:`~grgg.manifolds.Sphere`
        Can also be specified as an integer, which is interpreted as
        the dimension of a sphere with volume equal to `n_nodes`.
        Note that in this context manifold volume refers to the volume
        of the manifold surface, not of the enclosed space.
    layers
        Tuple of GRGG layers with different energy and coupling functions.
    """

    n_nodes: int = eqx.field(static=True)
    manifold: CompactManifold
    layers: tuple[AbstractLayer, ...]
    probability: ProbabilityFunction = eqx.field(repr=False)
    function: Callable[[Floats, Floats, Floats], Floats] = eqx.field(repr=False)

    def __init__(
        self,
        n_nodes: int,
        manifold: CompactManifold | int | tuple[int, type[CompactManifold]],
        layers: AbstractLayer | Sequence[AbstractLayer] = (),
        *more_layers: AbstractLayer,
        # quantizer: ArrayQuantizer | None = None,
        probability: ProbabilityFunction | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialization method.

        `**kwargs` are passed to the coupling function.
        """
        self.n_nodes = int(n_nodes)
        # self.quantizer = ArrayQuantizer() if quantizer is None else
        # Handle manifold initialization
        if not isinstance(layers, Iterable):
            layers = (layers,)
        layers = tuple(layers) + more_layers
        if layers and (
            isinstance(layers[0], int)
            or isinstance(layers[0], type)
            and not issubclass(layers[0], AbstractLayer)
        ):
            dim = layers[0]
            layers = layers[1:]
            self.manifold = self._make_manifold((dim, manifold))
        else:
            self.manifold = self._make_manifold(manifold)
        # Initialize functions
        self.probability = probability or ProbabilityFunction(
            CouplingFunction(self.manifold.dim, **kwargs)
        )
        self.function = self._define_function()
        # Initialize layers
        self.layers = tuple(layer.attach(self) for layer in layers)
        # Initialize integration and optimization namespaces
        # self.integrate = Integration(self)

    def __repr__(self) -> str:
        cn = self.__class__.__name__
        layers = ", ".join(repr(layer) for layer in self.layers)
        return f"{cn}({self.n_nodes}, {self.manifold}, {layers})"

    def __getitem__(
        self, index: int | slice | Iterable
    ) -> AbstractLayer | tuple[AbstractLayer, ...]:
        if isinstance(index, Iterable):
            layers = tuple(self.layers[i] for i in index)
        else:
            layers = [self.layers[index]]
        return self.replace(layers=tuple(layers))

    def __len__(self) -> int:
        return len(self.layers)

    def __call__(self, g: Floats, beta: Floats, mu: Floats) -> Floats:
        return self.function(g, beta, mu)

    def __add__(self, layer: AbstractLayer) -> Self:
        """Add a layer to the GRGG model using the `+` operator.

        Examples
        --------
        >>> from grgg import GRGG, Complementarity
        >>> GRGG(100, 1) + Complementarity()
        GRGG(100, Sphere(1, r=15.92), Complementarity(beta=f32[], mu=f32[], log=True))
        """
        return self.add_layer(layer)

    def n_units(self) -> int:
        """Number of units in the model."""
        return self.n_nodes

    @property
    def n_layers(self) -> int:
        """Number of layers in the model."""
        return len(self)

    @property
    def delta(self) -> float:
        """Sampling density."""
        return self.n_nodes / self.manifold.volume

    @property
    def is_quantized(self) -> bool:
        """Check if any of the model layers is quantized."""
        return False

    @property
    def is_heterogeneous(self) -> bool:
        """Check if any of the model layers is heterogeneous."""
        return any(layer.is_heterogeneous for layer in self.layers)

    @property
    def coupling(self) -> CouplingFunction:
        """The coupling function."""
        return self.probability.coupling

    @property
    def parameters(self) -> list[dict[str, jnp.ndarray]]:
        """Model parameters."""
        return [layer.parameters for layer in self.layers]

    def equals(self, other: object) -> bool:
        """Check if two models are equal."""
        return (
            super().equals(other)
            and self.n_nodes == other.n_nodes
            and self.manifold.equals(other.manifold)
            and len(self.layers) == len(other.layers)
            and all(
                l1.equals(l2) for l1, l2 in zip(self.layers, other.layers, strict=True)
            )
        )

    def add_layer(self, layer: AbstractLayer) -> Self:
        """Return a shallow copy with the new layer.

        Parameters
        ----------
        layer
            A GRGG layer to add.
        **constraints
            Soft constraints on the sufficient statistics.

        Examples
        --------
        Add a default complementarity layer and check that it is linked to the model.
        >>> from grgg import GRGG, Complementarity
        >>> model = GRGG(100, 1).add_layer(Complementarity())
        >>> model.layers[0].model is model
        True
        """
        if isinstance(layer, type) and issubclass(layer, AbstractLayer):
            layer = layer()
        if not isinstance(layer, AbstractLayer):
            errmsg = f"'layer' must be a '{AbstractLayer.__name__}' instance"
            raise TypeError(errmsg)
        layer = layer.attach(self)
        return self.replace(layers=(*self.layers, layer))

    def remove_layer(self, index: int) -> Self:
        """Return a shallow copy of the GRGG model without the specified layer.

        Parameters
        ----------
        index
            Index of the layer to remove.
        """
        layers = tuple(layer for i, layer in enumerate(self.layers) if i != index)
        return self.replace(layers=layers)

    def define_function(self) -> Callable[[Floats, Floats, Floats], Floats]:
        """Define the model function."""

        def model_function(g: Floats, beta: Floats, mu: Floats) -> Floats:
            """Compute the multilayer edge probabilities.

            Parameters
            ----------
            g
                Geodesic distances.
            beta
                Beta parameters.
            mu
                Mu parameters.
            """
            P = 1.0
            for i, layer in enumerate(self.layers):
                P *= 1 - layer.function(g, beta[i], mu[i])
            return 1 - P

        return model_function

    def sample(self, **kwargs: Any) -> "Sample":
        """Generate a model sample.

        See :meth:`~grgg.model._sampling.Sampler.sample` for details on `**kwargs`.
        """
        return self.nodes.sample(**kwargs)

    # Quantization methods ------------------------------------------------------------

    def dequantize(self) -> Self:
        """Dequantize model parameters."""
        return self

    # Internals ----------------------------------------------------------------------

    @singledispatchmethod
    def _make_manifold(self, manifold: CompactManifold) -> CompactManifold:
        if isinstance(manifold, tuple):
            manifold_type, dim = manifold
            if isinstance(manifold_type, int):
                manifold_type, dim = dim, manifold_type
            return self._make_manifold(dim, manifold_type)
        if not isinstance(manifold, CompactManifold):
            errmsg = "'manifold' must be a 'CompactManifold'"
            raise TypeError(errmsg)
        return manifold

    @_make_manifold.register
    def _(
        self, dim: int, manifold_type: type[CompactManifold] = Sphere
    ) -> CompactManifold:
        manifold = manifold_type(dim).with_volume(self.n_nodes)
        return self._make_manifold(manifold)

    @_make_manifold.register
    def _(self, dim: jnp.integer, *args: Any, **kwargs: Any) -> CompactManifold:
        return self._make_manifold(int(dim), *args, **kwargs)
