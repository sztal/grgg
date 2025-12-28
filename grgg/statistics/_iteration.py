from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from grgg._typing import DTypeLike, Integer, Integers, IntVector, Real, Reals
from grgg.abc import AbstractCallable
from grgg.utils.random import RandomGenerator

if TYPE_CHECKING:
    from grgg.models.base.ergm.model import AbstractErgm

__all__ = ("NodeIteration",)


NodeIterationKernelT = Callable[["AbstractErgm", Integer, ...], Real]
NodeIterationWeightsFunctionT = Callable[["AbstractErgm", int, Integers], Reals]


class NodeIteration(AbstractCallable):
    """Iterate a kernel function over node tuples rooted at specified focal node(s).

    Computes the sum of ``kernel(model, *focal_nodes, i, j, ...)`` over all node tuples
    of the form `(*focal_nodes, i, j, ...)`, where `focal_nodes` are fixed and the rest
    of indices are iterated.

    Parameters
    ----------
    n_nodes
        Number of nodes in the system.
    kernel
        Function taking ``(model, *focal_nodes, i, j, ...)`` and returning a scalar.
    weights
        Optional function taking ``(model, depth, *vids)`` and returning
        sampling weights for Monte Carlo mode. Required when ``mc > 0``.
    order
        Number of loops (i.e., number of iterated node indices).
    unique
        If ``True``, kernel returns 0 when indices are not all distinct.
        In MC mode, weights for already-used indices are zeroed before sampling.
    mc
        Number of Monte Carlo samples per depth level.
        If 0, exhaustive iteration is used.
    dtype
        Data type for the accumulator.
    key
        Default PRNG key for Monte Carlo sampling.

    Examples
    --------
    >>> import equinox as eqx
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph
    >>> from grgg.statistics.abc import NodeIteration

    Sum over pairs (order=1): kernel(model, i, j) for all j (including j=i)

    >>> model = RandomGraph(3)
    >>> @NodeIteration.from_kernel(order=1, unique=False)
    ... def sum_pairs(model, i, j):
    ...     return jnp.array(1.0)
    >>> sum_pairs(model, 0).item()  # 3 nodes to sum over
    3.0

    Sum over single node (order=0): just kernel(model, i)

    >>> @NodeIteration.from_kernel(order=0)
    ... def identity(model, i):
    ...     return jnp.array(5.0)
    >>> identity(model, 0).item()
    5.0

    Sum over triples (order=2): kernel(model, i, j, k) for all j, k

    >>> @NodeIteration.from_kernel(order=2, unique=False)
    ... def sum_triples(model, i, j, k):
    ...     return jnp.array(1.0)
    >>> sum_triples(model, 0).item()  # 3 * 3 = 9 pairs
    9.0

    With unique=True (default), only distinct index tuples contribute

    >>> @NodeIteration.from_kernel(order=1)
    ... def sum_pairs_unique(model, i, j):
    ...     return jnp.array(1.0)
    >>> sum_pairs_unique(model, 0).item()  # excludes (0, 0)
    2.0

    >>> @NodeIteration.from_kernel(order=2)
    ... def sum_triples_unique(model, i, j, k):
    ...     return jnp.array(1.0)
    >>> sum_triples_unique(model, 0).item()  # only (0,1,2) and (0,2,1)
    2.0

    Gradient computation: model as first arg enables clean autodiff

    >>> @NodeIteration.from_kernel(order=1, unique=False)
    ... def weighted_sum(model, i, j):
    ...     return model.mu * jnp.array(1.0)
    >>> grad_fn = eqx.filter_grad(weighted_sum)
    >>> grads = grad_fn(model, 0)
    >>> grads.mu.data.item()  # d/d(mu) of (mu * 3) = 3
    3.0
    """

    order: int = eqx.field(static=True, converter=int)
    kernel: NodeIterationKernelT = eqx.field(static=True)
    weights: NodeIterationWeightsFunctionT | None = eqx.field(static=True, default=None)
    unique: bool = eqx.field(static=True, default=True, kw_only=True)
    mc: int = eqx.field(static=True, default=0, kw_only=True, converter=int)
    key: Integers | None = eqx.field(default=None, kw_only=True)
    unroll: int = eqx.field(static=True, default=1, kw_only=True, converter=int)
    batch_size: int | None = eqx.field(static=True, default=None, kw_only=True)
    dtype: DTypeLike = eqx.field(static=True, default=float, kw_only=True)

    def __check_init__(self) -> None:
        if self.order < 0:
            errmsg = f"expected 'order' >= 0, got {self.order}"
            raise ValueError(errmsg)
        if self.mc < 0:
            errmsg = f"expected 'mc' >= 0, got {self.mc}"
            raise ValueError(errmsg)
        if self.mc > 0 and self.weights is None:
            errmsg = "MC sampling requires 'weights' function"
            raise ValueError(errmsg)

    @classmethod
    def from_kernel(
        cls, *args: Any, **kwargs: Any
    ) -> Callable[[NodeIterationKernelT], Self]:
        def wrapped(kernel: NodeIterationKernelT) -> Self:
            return cls(*args, kernel=kernel, **kwargs)

        return wrapped

    def map(
        self,
        model: "AbstractErgm",
        indices: Integers | None = None,
        **kwargs: Any,
    ) -> Reals:
        """Map the iteration over multiple focal nodes.

        Parameters
        ----------
        model
            The model to use for the iteration.
        indices
            The node indices to iterate over. If `None`, all nodes are used.
        **kwargs
            Additional keyword arguments passed to `jax.lax.map`.
        """
        if indices is None:
            indices = jnp.arange(model.n_nodes)
        if indices.ndim < 2:
            indices = indices[:, jnp.newaxis]
        kwargs = {"batch_size": self.batch_size, **kwargs}
        return jax.lax.map(
            eqx.filter_jit(lambda vids: self(model, vids, key=self.key)),
            indices,
            **kwargs,
        )

    def __call__(
        self,
        model: "AbstractErgm",
        focal_nodes: Integers,
        *,
        init: Real | None = None,
        key: Integers | None = None,
    ) -> Real:
        """Compute the sum of kernel over all node tuples rooted at focal nodes.

        Parameters
        ----------
        model
            The model to use for the iteration.
        focal_nodes
            Array of focal node indices that are fixed in the iteration.
        init
            Initial accumulator value.
        key
            Random key for MC sampling.

        Returns
        -------
        Real
            The sum of kernel evaluations over all node tuples.
        """
        focal_nodes = jnp.atleast_1d(jnp.asarray(focal_nodes))
        if init is None:
            init = jnp.array(0, dtype=self.dtype)
        # Ensure key is always a valid PRNG array (required by filter_custom_vjp)
        key = RandomGenerator.make_key(key)

        # Delegate to the custom VJP-wrapped function for memory-efficient gradients
        # Pack (self, model, init) as vjp_args since these may have gradients
        return _node_iteration_call(model, self, focal_nodes, init, key)

    def _call_exact(
        self,
        model: "AbstractErgm",
        focal_nodes: Integers,
        *,
        init: Real,
    ) -> Real:
        """Exact (exhaustive) iteration over all node tuples.

        Uses nested fori_loops with static depth unrolling.
        Memory: O(1) - only carry state is maintained.
        """
        n_nodes = model.n_nodes
        kernel = self.kernel
        unique = self.unique
        order = self.order
        unroll = self.unroll

        # Initialize vids array with first index set to i
        vids = jnp.full((order,), -1, dtype=int)

        # Build nested loops at trace time (Python recursion, not JAX recursion)
        def make_loop(depth: int) -> Callable:
            """Create a loop body for the given depth level."""
            if depth == order:
                # Base case: innermost body just applies the kernel
                def body_exact(
                    vid: Integer, value: tuple[Any, IntVector]
                ) -> tuple[Any, IntVector]:
                    carry, vids = value
                    vids = vids.at[order - 1].set(vid)
                    all_vids = jnp.concatenate([focal_nodes, vids])
                    result = kernel(model, *all_vids)
                    # Apply uniqueness mask
                    if unique:
                        is_unique = jnp.bool_(True)
                        for a in range(len(focal_nodes) + order):
                            for b in range(a + 1, len(focal_nodes) + order):
                                is_unique = is_unique & (all_vids[a] != all_vids[b])
                        result = _tree_mul(result, is_unique)
                    return (_tree_add(carry, result), vids)

                return body_exact

            # Recursive case: create a loop that nests another loop inside
            inner_loop = make_loop(depth + 1)

            def loop_exact(
                vid: Integer, value: tuple[Any, IntVector]
            ) -> tuple[Any, IntVector]:
                carry, vids = value
                vids = vids.at[depth - 1].set(vid)
                return jax.lax.fori_loop(
                    0,
                    n_nodes,
                    inner_loop,
                    init_val=(carry, vids),
                    unroll=unroll,
                )

            return loop_exact

        # Start from depth 1 (depth 0 is already set to i)
        loop = make_loop(1)
        result, _ = jax.lax.fori_loop(
            0,
            n_nodes,
            loop,
            init_val=(init, vids),
            unroll=unroll,
        )
        return result

    def _call_mc(
        self,
        model: "AbstractErgm",
        focal_nodes: Integers,
        *,
        init: Real,
        key: Integers | None,
    ) -> Real:
        """Monte Carlo iteration using importance sampling.

        Uses nested fori_loops with static depth unrolling.
        Memory: O(1) - fori_loop doesn't store intermediate states (unlike scan).

        The algorithm preserves the nested sampling structure of the original:
        - At each depth, sample `mc` indices and compute importance weight
        - Recurse to next depth for each sampled index
        - This creates correlated samples that can reduce variance
        """
        if key is None:
            errmsg = "MC sampling requires a PRNG key"
            raise ValueError(errmsg)

        order = self.order
        mc = self.mc
        unique = self.unique
        kernel = self.kernel
        weights = self.weights
        unroll = self.unroll
        n_nodes = model.n_nodes

        # Initialize vids array with first index set to i
        vids = jnp.full((order,), -1, dtype=int)

        # Build nested loops at trace time (static unrolling for depth)
        def make_loop(depth: int) -> Callable:
            """Create a loop body for the given depth level."""
            if depth == order:
                # Base case: innermost body just applies the kernel
                def body_mc_inner(
                    _sample_idx: Integer,
                    carry: tuple[Any, IntVector, Integers],
                ) -> tuple[Any, IntVector, Integers]:
                    result, vids, key = carry
                    # vids already has the sampled index set by outer loop
                    all_vids = jnp.concatenate([focal_nodes, vids])
                    contribution = kernel(model, *all_vids)
                    return (_tree_add(result, contribution), vids, key)

                return body_mc_inner

            # Recursive case: sample at this depth, then recurse
            inner_loop = make_loop(depth + 1)

            def loop_mc(
                sample_idx: Integer,
                carry: tuple[Any, IntVector, Integers],
            ) -> tuple[Any, IntVector, Integers]:
                result, vids, key = carry
                all_vids = jnp.concatenate([focal_nodes, vids])

                # Sample mc indices for next depth
                key = jax.random.fold_in(key, sample_idx)
                key, subkey = jax.random.split(key)

                # Compute weights for this depth
                w = weights(model, depth, all_vids)

                # Zero out already-used indices if unique
                # At depth d, vids[d] was just set by the outer loop.
                # We're sampling for position d+1. The original _sample_indices
                # with depth=d zeros range(d) = [0..d-1], excluding vids[d].
                if unique:
                    for a in range(depth):
                        w = w.at[all_vids[a]].set(0.0)

                # Compute importance sampling multiplier
                w_sum = w.sum()
                mult = w_sum / mc

                # Sample mc indices
                sampled = jax.random.choice(subkey, n_nodes, shape=(mc,), p=w / w_sum)

                # Create zero accumulator with same structure as init
                inner_init = jax.tree.map(jnp.zeros_like, init)

                # Iterate over sampled indices using fori_loop (not scan)
                def inner_body(
                    idx: Integer,
                    inner_carry: tuple[Any, IntVector, Integers],
                ) -> tuple[Any, IntVector, Integers]:
                    inner_result, inner_vids, inner_key = inner_carry
                    vid = sampled[idx]
                    inner_vids = inner_vids.at[depth].set(vid)
                    return inner_loop(idx, (inner_result, inner_vids, inner_key))

                inner_result, vids, key = jax.lax.fori_loop(
                    0,
                    mc,
                    inner_body,
                    init_val=(inner_init, vids, key),
                    unroll=unroll,
                )

                return (_tree_add(result, _tree_mul(inner_result, mult)), vids, key)

            return loop_mc

        # Sample at depth 0 and start recursion
        # Note: at depth 0, unique=True does NOT zero out the focal node
        # (matching original behavior where range(0) is empty)
        key, subkey = jax.random.split(key)
        all_vids = jnp.concatenate([focal_nodes, vids])
        w = weights(model, 0, all_vids)
        # No zeroing at depth 0 - unique only applies at depth >= 1
        w_sum = w.sum()
        mult = w_sum / mc
        sampled = jax.random.choice(subkey, n_nodes, shape=(mc,), p=w / w_sum)

        loop = make_loop(1)

        def outer_body(
            idx: Integer,
            carry: tuple[Any, IntVector, Integers],
        ) -> tuple[Any, IntVector, Integers]:
            result, vids, key = carry
            vid = sampled[idx]
            vids = vids.at[0].set(vid)
            return loop(idx, (result, vids, key))

        result, _, _ = jax.lax.fori_loop(
            0,
            mc,
            outer_body,
            init_val=(init, vids, key),
            unroll=unroll,
        )

        return _tree_mul(result, mult)


# Pytree accumulation helpers for NodeIteration
def _tree_add(a: Any, b: Any) -> Any:
    return jax.tree.map(jnp.add, a, b)


def _tree_mul(a: Any, s: Any) -> Any:
    return jax.tree.map(lambda x: x * s, a)


# Custom VJP for memory-efficient gradient computation in NodeIteration
@eqx.filter_custom_vjp
def _node_iteration_call(
    model: "AbstractErgm",
    iteration: NodeIteration,
    focal_nodes: Integers,
    init: Real,
    key: Integers,
) -> Real:
    """Core NodeIteration computation with custom VJP for memory efficiency."""
    if iteration.order == 0:
        return _tree_add(init, iteration.kernel(model, *focal_nodes))

    if iteration.mc > 0:
        return iteration._call_mc(model, focal_nodes, init=init, key=key)
    return iteration._call_exact(model, focal_nodes, init=init)


@_node_iteration_call.def_fwd
def _node_iteration_call_fwd(
    _perturbed: Any,
    model: "AbstractErgm",
    iteration: NodeIteration,
    focal_nodes: Integers,
    init: Real,
    key: Integers,
) -> tuple[Real, None]:
    """Forward pass: compute result, no residuals needed."""
    result = _node_iteration_call.fn(model, iteration, focal_nodes, init, key)
    return result, None


@_node_iteration_call.def_bwd
def _node_iteration_call_bwd(
    _residuals: None,
    g: Real,
    _perturbed: Any,
    model: "AbstractErgm",
    iteration: NodeIteration,
    focal_nodes: Integers,
    init: Real,  # noqa
    key: Integers,
) -> tuple[Any, None, None, Real, None]:
    """Backward pass: accumulate gradients using NodeIteration with grad kernel.

    Instead of storing all intermediate activations, we run another iteration
    with a "gradient kernel" that computes per-element VJPs. This gives
    O(kernel_size) memory instead of O(N × kernel_size).
    """
    original_kernel = iteration.kernel

    # Probe kernel output structure to create matching cotangent
    # We need a dummy call to get the output pytree structure
    dummy_vids = jnp.zeros((iteration.order + len(focal_nodes),), dtype=int)
    kernel_output_struct = jax.eval_shape(lambda: original_kernel(model, *dummy_vids))

    # Create gradient kernel: computes VJP of original kernel for each node tuple
    def grad_kernel(m: "AbstractErgm", *vids: Integer) -> Any:
        def fwd(m: "AbstractErgm") -> Any:
            return original_kernel(m, *vids)

        _, vjp_fn = eqx.filter_vjp(fwd, m)
        cotangent = jax.tree.map(
            lambda x: jnp.full(x.shape, g, x.dtype), kernel_output_struct
        )
        (grad_m,) = vjp_fn(cotangent)
        return grad_m

    # Probe grad_kernel output structure for proper initialization
    # filter_vjp returns gradients with same structure as eqx.filter(model, is_inexact)
    grad_structure = eqx.filter(model, eqx.is_inexact_array)
    grad_init = jax.tree.map(jnp.zeros_like, grad_structure)

    # Create and run new iteration with gradient kernel (reuses all iteration machinery)
    grad_iteration = iteration.replace(kernel=grad_kernel)
    grad_model = grad_iteration(model, focal_nodes, init=grad_init, key=key)
    return grad_model
