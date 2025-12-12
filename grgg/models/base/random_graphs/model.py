from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import track
from scipy.sparse import csr_array

from grgg._options import options
from grgg._typing import Booleans, BoolVector, Integer, IntVector, Real
from grgg.models.base.ergm import AbstractErgm, ErgmSample
from grgg.utils.compute import fori
from grgg.utils.dispatch import dispatch
from grgg.utils.misc import batch_starts
from grgg.utils.random import RandomGenerator

from .functions import AbstractRandomGraphFunctions
from .views import AbstractRandomGraphNodePairView, AbstractRandomGraphNodeView

__all__ = ("AbstractRandomGraph",)


class AbstractRandomGraph(AbstractErgm):
    """Abstract base class for random graph models."""

    functions: eqx.AbstractClassVar[type[AbstractRandomGraphFunctions]]

    nodes_cls: eqx.AbstractClassVar[type[AbstractRandomGraphNodeView]]
    pairs_cls: eqx.AbstractClassVar[type[AbstractRandomGraphNodePairView]]

    # Model functions ----------------------------------------------------------------

    def free_energy(self, *args: Any, **kwargs: Any) -> Real:
        """Compute the free energy of the model."""
        fe = self._free_energy(*args, **kwargs)
        return fe / 2 if self.is_undirected else fe

    def _free_energy(self, *args: Any, **kwargs: Any) -> Real:
        """Implementation of free energy function."""
        if self.is_homogeneous:
            return _free_energy_homogeneous(self, *args, **kwargs)
        return _free_energy_heterogeneous(self, *args, **kwargs)

    # Model fitting interface --------------------------------------------------------

    @dispatch
    def get_default_fit_method(self: "AbstractRandomGraph", data: Any) -> str:  # noqa
        """Get the default fitting method for a given model and target statistics."""
        return "lagrangian"

    # Model sampling interface -------------------------------------------------------

    def sample(
        self,
        *,
        batch_size: int | None = None,
        rng: RandomGenerator | int | None = None,
        progress: bool | Mapping | None = None,
    ) -> ErgmSample:
        """Sample a graph from the model.

        Parameters
        ----------
        batch_size
            Batch size for processing node pairs.
            If less than or equal to 0, process all node pairs at once.
        rngs
            Random state for reproducibility, can be an integer seed,
            a `nnx.Rngs` object, or `None` for random initialization.
        progress
            Whether to display a progress bar. Can be a boolean or a
            dictionary of keyword arguments passed to `tqdm.tqdm`.
        """
        i, j = (
            np.asarray(k)
            for k in _sample_impl(
                self,
                batch_size=batch_size,
                rng=rng,
                progress=progress,
            )
        )
        n_nodes = self.model.nodes.n_nodes
        A = csr_array((np.ones_like(i), (i, j)), shape=(n_nodes, n_nodes))
        if not self.model.is_directed:
            A = A + A.T  # make symmetric
        return ErgmSample(A)


# Implementation ---------------------------------------------------------------------


def _sample_impl(
    model: AbstractRandomGraph,
    *,
    batch_size: int | None = None,
    rng: RandomGenerator | int | None = None,
    progress: bool | Mapping | None = None,
) -> tuple[IntVector, IntVector]:
    batch_size = options.sampling.get("batch_size", batch_size)
    rng = RandomGenerator.from_seed(rng)
    n_nodes = model.nodes.n_nodes
    starts = batch_starts(n_nodes, batch_size, repeat=2)
    starts = starts[starts[:, 0] <= starts[:, 1]]
    Ai = []
    Aj = []
    progress_opts = options.progress.from_steps(
        len(starts), progress, description="Sampling..."
    )
    for s1, s2 in track(starts, **progress_opts):
        if not model.is_directed and s1 > s2:
            continue
        if s1 == s2:
            bs = int(min(batch_size, n_nodes - s1))
            i, j, M = _sample_impl_diag(model, s1, bs, rng)
            i = i[M]
            j = j[M]
        else:
            bs1 = int(min(batch_size, n_nodes - s1))
            bs2 = int(min(batch_size, n_nodes - s2))
            M = _sample_impl_offdiag(model, s1, s2, bs1, bs2, rng)
            i, j = jnp.where(M)
            i += s1
            j += s2
        Ai.append(i)
        Aj.append(j)
    Ai = jnp.concatenate(Ai)
    Aj = jnp.concatenate(Aj)
    return Ai, Aj


@eqx.filter_jit
def _sample_impl_diag(
    model: AbstractRandomGraph,
    s: Integer,
    batch_size: int,
    rng: RandomGenerator,
) -> tuple[IntVector, IntVector, BoolVector]:
    """Sample edges for the diagonal block."""
    i, j = jnp.triu_indices(batch_size, k=1)
    i += s
    j += s
    p = model.pairs[i, j].probs()
    mask = rng.uniform(shape=p.shape) < p
    return i, j, mask


@eqx.filter_jit
def _sample_impl_offdiag(
    model: AbstractRandomGraph,
    s1: Integer,
    s2: Integer,
    batch_size1: int,
    batch_size2: int,
    rng: RandomGenerator,
) -> Booleans:
    """Sample edges for the off-diagonal block."""
    i = jnp.arange(batch_size1) + s1
    j = jnp.arange(batch_size2) + s2
    p = model.pairs[jnp.ix_(i, j)].probs()
    mask = rng.uniform(shape=p.shape) < p
    return mask


@eqx.filter_jit
def _free_energy_homogeneous(
    model: "AbstractRandomGraph", *args: Any, normalize: bool = False, **kwargs: Any
) -> Real:
    """Compute the free energy of a homogeneous model."""
    fe = model.nodes.free_energy(*args, **kwargs)
    return fe if normalize else fe * model.n_nodes


@eqx.filter_custom_vjp
@eqx.filter_jit
def _free_energy_heterogeneous(
    model: "AbstractRandomGraph", *args: Any, **kwargs: Any
) -> Real:
    """Compute the free energy of a heterogeneous model."""

    @fori(0, model.n_nodes, init=0.0)
    def fe(i: Integer, carry: Real) -> Real:
        fe = model.functions.node_free_energy(model, i, *args, **kwargs)
        return carry + fe

    return fe


@_free_energy_heterogeneous.def_fwd
@eqx.filter_jit
def _free_energy_heterogeneous_fwd(
    _, model: "AbstractRandomGraph", *args: Any, **kwargs: Any
) -> tuple[Real, None]:
    """Forward pass for custom VJP of free energy."""
    return _free_energy_heterogeneous(model, *args, **kwargs), None


@_free_energy_heterogeneous.def_bwd
@eqx.filter_jit
def _free_energy_heterogeneous_bwd(
    _,
    g_out: Real,
    __,
    model: "AbstractRandomGraph",
    *args: Any,
    **kwargs: Any,
) -> "AbstractRandomGraph":
    """Backward pass for custom VJP of free energy."""
    # Initialize gradients with zeros matching the model structure
    init_grads = jax.tree_util.tree_map(jnp.zeros_like, model)
    # Pre-compile the gradient function for a single index
    grad_fn = eqx.filter_grad(model.functions.node_free_energy)

    @fori(0, model.n_nodes, init=init_grads)
    def gradient(i: Integer, carry: "AbstractRandomGraph") -> "AbstractRandomGraph":
        # Compute gradient for the i-th pair/node
        g_i = grad_fn(model, i, *args, **kwargs)
        # Accumulate gradients
        return jax.tree_util.tree_map(jnp.add, carry, g_i)

    # Apply the chain rule:
    # multiply accumulated grads by the output gradient (g_out)
    return jax.tree_util.tree_map(lambda g: g * g_out, gradient)
