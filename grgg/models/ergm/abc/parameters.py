import equinox as eqx

from grgg.models.abc import AbstractParameter

__all__ = ("AbstractErgmParameter",)


class AbstractErgmParameter(AbstractParameter):
    """Abstract base class for ERGM parameters."""

    sufficient_statistic: eqx.AbstractClassVar[str]
