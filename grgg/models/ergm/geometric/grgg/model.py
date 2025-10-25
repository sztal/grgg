from collections.abc import Iterable, Sequence
from functools import singledispatchmethod
from typing import Any, Self

import equinox as eqx
import jax.numpy as jnp

from grgg._options import options
from grgg._typing import Reals
from grgg.manifolds import CompactManifold, Sphere
from grgg.models.ergm.random_graphs.undirected import RandomGraph

from .functions import (
    CouplingFunction,
    ProbabilityFunction,
)
from .layers import AbstractLayer
from .parameters import ParameterGroups


class GRGG(RandomGraph, Sequence[Self]):
    r"""Generalized Random Geometric Graph model.

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

    Examples
    --------
    Create a homogeneous similarity-driven GRGG model with 100 nodes on a 2D sphere.
    For simplicity, we will use the non-logarithmic energies and non-modified couplings.
    >>> from grgg import GRGG, Similarity, options
    >>> options.model.log = False
    >>> options.model.modified = False
    >>> model = GRGG(100, 2) + Similarity()
    >>> model
    GRGG(100, Sphere(2, r=2.82), Similarity(mu=f32[], beta=f32[], log=False))
    >>> model.n_layers
    1
    >>> model.n_nodes
    100
    >>> model.is_heterogeneous  # model is node-homogeneous
    False

    Now, we will use it to compute edge probability at a given distance.
    We are using raw energies, so since also by default :math:`\mu = 0`
    we must get 1/2 prob for zero distance and something that should be very close
    to zero at the maximal distance, known as the manifold diameter.
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> g = jnp.array([0, model.manifold.diameter])
    >>> p = model.pairs.probs(g)
    >>> p.tolist()
    [0.5, 0.0]

    Before going further, let us emphasize the crucial point that the model is
    implemented using :mod:`jax` and is therefore jit-compilable and differentiable.
    Let us see how this works by computing the gradients of the edge probabilities.
    >>> import jax
    >>> # By default jax differentiates with respect to the first argument,
    >>> # so we put the model first
    >>> @jax.jit
    ... def probs(model, g): return model.pairs.probs(g)
    >>> # Check that the jit compilation works
    >>> probs(model, g).tolist()
    [0.5, 0.0]

    Now we compute the full jacobian (matrix of gradients at the two distances).
    >>> probs_jac = jax.jacobian(probs)
    >>> deriv = probs_jac(model, g)
    >>> deriv
    GRGG(100, Sphere(2, r=2.82), Similarity(mu=f32[2], beta=f32[2], log=False))

    Looks weird, but it is correct; :mod:`jax` returns the same structure as the input,
    now with gradient arrays in place of the original scalar parameters
    We can see better by extracting parameter containers from the model.
    >>> np.asarray(deriv.parameters[0].beta)  # wrap in numpy to make the doctest work
    array([-4.9999999e-10, -5.0358325e-11], dtype=...)
    >>> np.asarray(deriv.parameters[0].mu)
    array([7.500000e-01, 8.523493e-12], dtype=...)

    More generally, the parameters (or computed derivatives etc.) are neatly organized
    in parameter containers.
    >>> deriv.parameters
    ParameterGroups(
        Parameters(mu=f32[2], beta=f32[2])
        weights=f32[]
    )

    Now, let us extract a more explicit edge probability function from the model,
    which we can use to check the gradients using :mod:`scipy` and finite differences.
    >>> # extract model parameters
    >>> mu, beta = model.parameters[0].values()
    >>> np.asarray(model.probability(g, mu, beta))
    array([5.0000000e-01, 2.8411642e-12], dtype=...)

    Now we can check finite differences using scipy.
    >>> from scipy.differentiate import jacobian
    >>> params = np.array([mu, beta])
    >>> jac_g0 = jacobian(lambda x: model.probability(g[0], *x), params)
    >>> jac_g1 = jacobian(lambda x: model.probability(g[1], *x), params)
    >>> jac_fd = np.stack([jac_g0.df, jac_g1.df])
    >>> jac_fd
    array([[ 7.500019e-01,  0.000000e+00],
           [ 8.523494e-12, -5.035830e-11]], ...)

    So indeed, our jacobian computation is correct.
    >>> jnp.allclose(deriv.parameters.array, jac_fd).item()
    True

    Reset model options to default values.
    >>> options.reset()
    """

    n_nodes: int = eqx.field(static=True)
    manifold: CompactManifold
    layers: tuple[AbstractLayer, ...]
    probability: ProbabilityFunction = eqx.field(repr=False)
    eps: float = eqx.field(static=True, repr=False)

    def __init__(
        self,
        n_nodes: int,
        manifold: CompactManifold | int | tuple[int, type[CompactManifold]],
        layers: AbstractLayer | Sequence[AbstractLayer] = (),
        *more_layers: AbstractLayer,
        probability: ProbabilityFunction | None = None,
        eps: float | None = None,
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
        # self.integrate = Integration(self)
        # Initialize layers
        layers = [layer() if isinstance(layer, type) else layer for layer in layers]
        self.layers = tuple(layer.detach().attach(self) for layer in layers)
        # Other settings
        self.eps = float(options.model.eps if eps is None else eps)

    def __check_init__(self) -> None:
        for layer in self.layers:
            if layer._model_getter is None or layer.model is not self:
                errmsg = "all layers must be linked to the model"
                raise ValueError(errmsg)
        if self.eps <= 0:
            errmsg = "'eps' must be positive"
            raise ValueError(errmsg)

    def _repr_inner(self) -> str:
        inner = f"{super()._repr_inner()}, {self.manifold}"
        layers = ", ".join(repr(layer) for layer in self.layers)
        if layers:
            inner += f", {layers}"
        return inner

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

    def __call__(self, g: Reals, mu: Reals, beta: Reals) -> Reals:
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
        g, mu, beta = self._preprocess_inputs(g, mu, beta)
        P = 1.0
        for i, layer in enumerate(self.layers):
            P *= 1 - layer(g, mu[i], beta[i])
        return 1 - P

    def __add__(self, layer: AbstractLayer) -> Self:
        """Add a layer to the GRGG model using the `+` operator.

        Examples
        --------
        >>> from grgg import GRGG, Complementarity
        >>> GRGG(100, 1) + Complementarity()
        GRGG(100, Sphere(1, r=15.92), Complementarity(mu=f32[], beta=f32[], log=True))
        """
        return self.add_layer(layer)

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
    def parameters(self) -> ParameterGroups:
        """Model parameters."""
        groups = [layer.parameters for layer in self.layers]
        if self.is_quantized:
            return ParameterGroups(groups, weights=self.quantizer.counts)
        return ParameterGroups(groups)

    def _equals(self, other: object) -> bool:
        """Check if two models are equal."""
        return (
            super()._equals(other)
            and self.n_nodes == other.n_nodes
            and self.manifold.equals(other.manifold)
            and self.probability.equals(other.probability)
            and self.eps == other.eps
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
        >>> model = GRGG(100, 1).add_layer(Complementarity)
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

    def set_parameters(
        self, parameters: Iterable[Any] = (), *more_parameters: Any
    ) -> Self:
        """Get shallow copy with update parameter values.

        Parameters
        ----------
        parameters
            An iterable of parameter dictionaries, one for each layer.
        more_parameters
            Additional parameter dictionaries.

        Examples
        --------
        Update the parameters of a two-layer model.
        >>> from grgg import GRGG, Complementarity, Similarity
        >>> model = (
        ...     GRGG(100, 1) +
        ...     Complementarity() +
        ...     Similarity()
        ... )
        >>> new_params = [
        ...     {"mu": 0.1, "beta": 0.5},
        ...     {"mu": 0.2, "beta": 1.0},
        ... ]
        >>> updated_model = model.set_parameters(new_params)
        >>> updated_model.layers[0].mu.item(), updated_model.layers[0].beta.item()
        (0.1, 0.5)
        >>> updated_model.layers[1].mu.item(), updated_model.layers[1].beta.item()
        (0.2, 1.0)
        >>> updated_model.equals(model)
        False
        """
        parameters = tuple(parameters) + more_parameters
        if len(parameters) != len(self.layers):
            errmsg = (
                "number of parameter sets must match number of layers; "
                "pass empty dicts for layers that should not be updated"
            )
            raise ValueError(errmsg)
        if not parameters:
            return self
        layers = [
            layer.set_parameters(p)
            for layer, p in zip(self.layers, parameters, strict=True)
        ]
        return self.replace(layers=tuple(layers))

    # Quantization methods ------------------------------------------------------------

    def quantize(self, n_codes: int | None = None, **kwargs: Any) -> Self:
        """Quantize node parameters.

        Parameters
        ----------
        n_codes
            Number of quantization codes.
        **kwargs
            Additional keyword arguments passed to
            :meth:`~grgg.quantize.ArrayQuantizer.from_data`.

        Examples
        --------
        Quantize a model with 1000 nodes to 100 units.
        >>> from grgg import GRGG, Complementarity, Similarity, RandomGenerator
        >>> rng = RandomGenerator(17)
        >>> n = 1000
        >>> model = (
        ...     GRGG(n, 1) +
        ...     Complementarity(rng.normal(n), rng.normal(n)**2) +
        ...     Similarity(rng.normal(n), rng.normal(n)**2)
        ... )
        >>> quantized = model.quantize(n_codes=100)
        >>> quantized
        QuantizedGRGG(100, Sphere(1, r=159.15), ...)
        >>> quantized.manifold.volume  # the manifold is not changed
        1000.0
        >>> quantized.n_nodes  # the true number of nodes is unchanged
        1000
        >>> quantized.n_units  # but the number of units is now 100
        100
        >>> # quantized parameters have weights equal to the bin counts
        >>> quantized.parameters.weights.size == quantized.n_units
        True
        >>> quantized.equals(model)
        False
        >>> dequantized = quantized.dequantize()
        >>> dequantized.equals(model)  # dequantization is lossy by design
        False
        >>> dequantized.n_units == model.n_units
        True
        """
        from grgg.models.geometric.grgg.quantized import QuantizedGRGG

        if self.is_quantized:
            errmsg = "model is already quantized"
            raise ValueError(errmsg)
        if self.is_homogeneous:
            return self
        if n_codes is not None:
            kwargs["n_codes"] = n_codes
        return QuantizedGRGG.from_model(self, **kwargs)

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
