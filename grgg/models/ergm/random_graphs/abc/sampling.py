from collections.abc import Mapping
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from rich.progress import track
from scipy.sparse import csr_array

from grgg._options import options
from grgg._typing import Booleans, BoolVector, Integer, IntVector
from grgg.models.ergm.abc import AbstractErgmSampler, ErgmSample
from grgg.utils.misc import batch_starts
from grgg.utils.random import RandomGenerator

if TYPE_CHECKING:
    from .model import AbstractRandomGraph

__all__ = ("AbstractRandomGraphSampler",)


T = TypeVar("T", bound="AbstractRandomGraph")


class AbstractRandomGraphSampler[T](AbstractErgmSampler[T]):
    """Abstract base class for samplers of random graph models."""

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
        i, j = self._sample(
            batch_size=batch_size,
            rng=rng,
            progress=progress,
        )
        i, j = tuple(np.asarray(k) for k in (i, j))
        n_nodes = self.nodes.n_nodes
        A = csr_array((np.ones_like(i), (i, j)), shape=(n_nodes, n_nodes))
        A = A + A.T  # make symmetric
        return ErgmSample(A)

    def _sample(
        self,
        *,
        batch_size: int | None = None,
        rng: RandomGenerator | int | None = None,
        progress: bool | Mapping | None = None,
    ) -> tuple[IntVector, IntVector]:
        if self.nodes.is_active:
            nodes = self.nodes.materialize(copy=False).nodes
            self = nodes.sampler
        if batch_size is None:
            batch_size = options.loop.batch_size or self.model.n_nodes
        rng = RandomGenerator.from_seed(rng)
        n_nodes = self.nodes.n_nodes
        starts = batch_starts(n_nodes, batch_size, repeat=2)
        starts = starts[starts[:, 0] <= starts[:, 1]]
        Ai = []
        Aj = []
        progress_opts = options.progress.from_steps(
            len(starts), progress, description="Sampling..."
        )
        for s1, s2 in track(starts, **progress_opts):
            if s1 > s2:
                continue
            if s1 == s2:
                bs = int(min(batch_size, n_nodes - s1))
                i, j, M = self._sample_diag(s1, bs, rng)
                i = i[M]
                j = j[M]
            else:
                bs1 = int(min(batch_size, n_nodes - s1))
                bs2 = int(min(batch_size, n_nodes - s2))
                M = self._sample_offdiag(s1, s2, bs1, bs2, rng)
                i, j = jnp.where(M)
                i += s1
                j += s2
            Ai.append(i)
            Aj.append(j)
        Ai = jnp.concatenate(Ai)
        Aj = jnp.concatenate(Aj)
        return Ai, Aj

    def _sample_diag(
        self, *args: Any, **kwargs: Any
    ) -> tuple[IntVector, IntVector, BoolVector]:
        return _sample_diag(self, *args, **kwargs)

    def _sample_offdiag(self, *args: Any, **kwargs: Any) -> Booleans:
        return _sample_offdiag(self, *args, **kwargs)


# Internals --------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("batch_size",))
def _sample_diag(
    sampler: AbstractRandomGraphSampler,
    s: Integer,
    batch_size: int,
    rng: RandomGenerator,
) -> tuple[IntVector, IntVector, BoolVector]:
    """Sample edges for the diagonal block."""
    i, j = jnp.triu_indices(batch_size, k=1)
    i += s
    j += s
    p = sampler.model.pairs[i, j].probs()
    mask = rng.uniform(shape=p.shape) < p
    return i, j, mask


@partial(jax.jit, static_argnames=("batch_size1", "batch_size2"))
def _sample_offdiag(
    sampler: AbstractRandomGraphSampler,
    s1: Integer,
    s2: Integer,
    batch_size1: int,
    batch_size2: int,
    rng: RandomGenerator,
) -> Booleans:
    """Sample edges for the off-diagonal block."""
    i = jnp.arange(batch_size1) + s1
    j = jnp.arange(batch_size2) + s2
    p = sampler.model.pairs[jnp.ix_(i, j)].probs()
    mask = rng.uniform(shape=p.shape) < p
    return mask
