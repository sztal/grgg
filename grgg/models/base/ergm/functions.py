from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from grgg._typing import Real, Reals

from ..model import AbstractModelFunctions

if TYPE_CHECKING:
    from .model import AbstractErgm

__all__ = ("AbstractErgmFunctions",)


class AbstractErgmFunctions(AbstractModelFunctions):
    """ERGM model functions container.

    Examples
    --------
    One can check that Hamiltonian and partition function can be used to calculate
    graph probabilities.
    >>> import jax.numpy as jnp
    >>> from grgg import RandomGraph, RandomGenerator
    >>> rng = RandomGenerator(303)
    >>> n = 100
    >>> model = RandomGraph(n, mu=rng.normal(n) - 1.5)
    >>> S = model.sample(rng=rng)
    >>> A = S.A.toarray()
    >>> D = A.sum(axis=1)
    >>> i, j = jnp.tril_indices(n, k=-1)
    >>> P = model.pairs[i, j].probs()
    >>> A = A[i, j]
    >>> loglik1 = jnp.log(P[A == 1]).sum() + jnp.log((1 - P)[A == 0]).sum()
    >>> H = model.hamiltonian(S.A)
    >>> F = model.free_energy()
    >>> loglik2 = -H + F
    >>> jnp.isclose(loglik1, loglik2).item()
    True
    """

    @classmethod
    def free_energy(
        cls, model: "AbstractErgm", *args: Any, normalize: bool = True, **kwargs: Any
    ) -> Reals:
        """Compute the free energy of the model.

        Parameters
        ----------
        model
            The model to compute the free energy for.
        normalize
            Whether to normalize the free energy by the number of nodes.
        *args, **kwargs
            Additional arguments.
        """
        raise NotImplementedError

    @classmethod
    def partition_function(
        cls, model: "AbstractErgm", *args: Any, **kwargs: Any
    ) -> Reals:
        """Compute the partition function of the model.

        Parameters
        ----------
        model
            The model to compute the partition function for.
        *args, **kwargs
            Additional arguments passed to :meth:`free_energy`.
        """
        free_energy = cls.free_energy(model, *args, **kwargs)
        return jnp.exp(-free_energy)

    @classmethod
    def hamiltonian(
        cls, model: "AbstractErgm", obj: Any, *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the Hamiltonian of the model.

        Parameters
        ----------
        model
            The model to compute the Hamiltonian for.
        obj
            The object (e.g., graph) to compute the Hamiltonian of.
        *args, **kwargs
            Passed to :meth:`grgg.models.ergm.AbstractErgm.fit`.
        """
        fit = model.fit(obj, *args, **{"every": False, **kwargs})
        return fit.hamiltonian()

    @classmethod
    def lagrangian(
        cls, model: "AbstractErgm", obj: Any, *args: Any, **kwargs: Any
    ) -> Real:
        """Compute the Lagrangian of the model.

        Parameters
        ----------
        model
            The model to compute the Lagrangian for.
        obj
            The object (e.g., graph) to compute the Lagrangian of.
        *args, **kwargs
            Passed to :meth:`grgg.models.ergm.AbstractErgm.fit`.
        """
        fit = model.fit(obj, *args, **{"every": False, **kwargs})
        H = fit.hamiltonian()
        F = model.free_energy()
        return H - F
