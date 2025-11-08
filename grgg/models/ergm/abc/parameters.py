import equinox as eqx

from grgg.models.abc import AbstractParameter
from grgg.statistics.abc import AbstractErgmStatistic

__all__ = ("AbstractErgmParameter",)


class AbstractErgmParameter(AbstractParameter):
    """Abstract base class for ERGM parameters."""

    statistic: eqx.AbstractClassVar[type[AbstractErgmStatistic]]
