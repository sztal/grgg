from typing import TYPE_CHECKING, TypeVar

import equinox as eqx

from grgg.abc import AbstractModule

if TYPE_CHECKING:
    from .models import AbstractModel

    T = TypeVar("T", bound=AbstractModel)


class AbstractModelModule[T](AbstractModule):
    """Abstract base class for model modules."""

    model: eqx.AbstractVar["AbstractModel"]
