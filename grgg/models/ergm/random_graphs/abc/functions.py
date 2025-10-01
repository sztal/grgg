from abc import abstractmethod
from typing import Any

from grgg._typing import Reals
from grgg.abc import AbstractFunction

__all__ = ("AbstractCoupling",)


class AbstractCoupling(AbstractFunction):
    """Abstract base class for coupling functions."""

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Reals:
        """Evaluate the coupling function."""
