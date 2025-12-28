from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any, ClassVar, TypeVar

import equinox as eqx
import lineax as lx
import optimistix as optx

from grgg._options import options
from grgg._typing import ObjectiveFunction
from grgg.utils.dispatch import dispatch
from grgg.utils.misc import split_kwargs
from grgg.utils.variables import ArrayBundle

from .modules import AbstractModelModule

__all__ = ("AbstractModelFit", "LeastSquaresFit")


T = TypeVar("T", bound="AbstractModel")
O = TypeVar("O", bound="ArrayBundle")

SolverMethodT = Callable[[ObjectiveFunction, ...], optx.Solution]


class AbstractModelFit[T, O](AbstractModelModule[T]):
    """Abstract base class for model fitting procedures."""

    model: eqx.AbstractVar[T]
    target: eqx.AbstractVar[O]

    is_fitted: bool = eqx.field(default=False, static=True, kw_only=True)
    is_initialized: bool = eqx.field(default=False, static=True, kw_only=True)

    method: eqx.AbstractClassVar[str]
    solver_cls: eqx.AbstractClassVar[type[eqx.Module]]

    def __check_init__(self) -> None:
        expected_stat_names = [
            p.get_statistic(self.model, self.method, homogeneous=False)[0]
            for p in self.model.parameters
        ]
        missing_stats = set(expected_stat_names) - set(self.target.mapping.keys())
        if missing_stats:
            errmsg = (
                f"target is missing required statistics: "
                f"{', '.join(repr(s) for s in missing_stats)}."
            )
            raise ValueError(errmsg)

    @abstractmethod
    def define_objective(self, *args: Any, **kwargs: Any) -> ObjectiveFunction:
        """Define the objective function for fitting."""

    def compute_expectations(
        self,
        model: T | None = None,
        *,
        normalize: bool = False,
        options: Mapping[str, Any] | None = None,
        **stats_options: Mapping[str, Any],
    ) -> ArrayBundle:
        """Compute expected target values.

        Parameters
        ----------
        model
            The model to compute expectations for. If `None`, uses `self.model`.
        normalize
            Whether to normalize the expectations by the number of units.
        options
            General options to pass to all statistic methods.
        **stats_options
            Specific options to pass to individual statistic methods.
        """
        model = model if model is not None else self.model
        options = options or {}
        expectations = {}
        for name, param in model.parameters.mapping.items():
            stat, method = param.get_statistic(model, self.method)
            opts, _ = split_kwargs(
                method.get_instance_fields(),
                **{**options, **stats_options.get(name, {}), "normalize": normalize},
            )
            expectation = method(**opts)
            expectations[stat] = expectation
        return ArrayBundle(**expectations)

    def get_expected_statistics_names(self) -> list[str]:
        """Get the names of the expected statistics used in fitting."""
        return [
            param.get_statistic(self.model, self.method)[0]
            for param in self.model.parameters
        ]

    @dispatch
    def get_tags(self, model: Any) -> frozenset:  # noqa
        """Get :mod:`lineax` tags describing the structure of the objective Hessian.

        Parameters
        ----------
        model
            The model to get tags for.
        """
        return frozenset()  # Default: no special tags

    @dispatch
    def get_solver_method(self, solver: eqx.Module) -> SolverMethodT:  # noqa
        """Get solver method for fit optimization."""
        return optx.minimise

    @get_solver_method.dispatch
    def _(self, solver: optx.AbstractLeastSquaresSolver) -> SolverMethodT:  # noqa
        """Get solver method for least squares solvers."""
        return optx.least_squares

    @get_solver_method.dispatch
    def _(self, solver: optx.AbstractFixedPointSolver) -> SolverMethodT:  # noqa
        """Get solver method for fixed point solvers."""
        return optx.fixed_point

    @get_solver_method.dispatch
    def _(self, solver: optx.AbstractRootFinder) -> SolverMethodT:  # noqa
        """Get solver method for root finding solvers."""
        return optx.root_find

    def _get_solver(
        self,
        solver: eqx.Module | type[eqx.Module] | None = None,
        **kwargs: Any,
    ) -> eqx.Module:
        """Get an instance of the solver for optimization.

        Parameters
        ----------
        solver
            The solver class or instance to use. If `None`, uses `self.solver_cls`.
        **kwargs
            Additional keyword arguments to pass to the solver constructor, or used
            to replace fields if `solver` is an instance.

        Returns
        -------
        An instance of the solver.
        """
        if isinstance(solver, eqx.Module):
            if kwargs:
                solver = replace(solver, **kwargs)
            return solver
        if solver is None:
            solver_cls = self.solver_cls
        # Merge options from global settings
        opts = {**options.solver.options, **kwargs}
        return solver_cls(**opts)

    def solve(
        self,
        *,
        objective: ObjectiveFunction | Mapping[str, Any] | None = None,
        solver: eqx.Module | type[eqx.Module] | None = None,
        **kwargs: Any,
    ) -> optx.Solution:
        """Solve the optimization problem.

        Parameters
        ----------
        objective
            The objective function to minimize. If a mapping is provided, it is used
            as keyword arguments to `self.define_objective()`. If `None`, uses
            `self.define_objective()`.
        solver
            The solver class or instance to use. If `None`, uses `self.solver_cls`.
        **kwargs
            Additional keyword arguments to pass to the solver constructor.

        Returns
        -------
        The solution of the optimization.
        """
        if isinstance(objective, Mapping):
            objective = self.define_objective(**objective)
        elif objective is None:
            objective = self.define_objective()
        solver = self._get_solver(solver)
        control = {
            "tags": self.get_tags(self.model),
            **options.solver.control.replace(**kwargs).to_dict(),
        }
        solver_method = self.get_solver_method(solver)
        solution = solver_method(
            fn=objective,
            solver=solver,
            y0=self.model,
            **control,
        )
        return solution


class LeastSquaresFit(AbstractModelFit[T, O]):
    """Least squares fitting procedure."""

    model: T
    target: O
    method: ClassVar[str] = "least_squares"
    solver_cls: ClassVar[type[eqx.Module]] = optx.LevenbergMarquardt

    @dispatch
    def get_tags(self, model: Any) -> frozenset:  # noqa
        return frozenset({lx.symmetric_tag})


# Avoid circular imports -------------------------------------------------------------

from .model import AbstractModel  # noqa
