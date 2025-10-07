from typing import ClassVar

import equinox as eqx

from grgg.models.abc import AbstractParameters
from grgg.models.ergm.random_graphs.abc import Mu

__all__ = ("UndirectedRandomGraphParameters",)


class UndirectedRandomGraphParameters(AbstractParameters):
    """Parameters for undirected random graph models."""

    parameters: tuple[Mu] = eqx.field(default_factory=lambda: (Mu(),))
    definition: ClassVar[tuple[type[Mu]]] = (Mu,)

    def _check_names(self) -> None:
        return self.names == ("mu",)
