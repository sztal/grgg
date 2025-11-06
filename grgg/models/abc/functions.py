from .modules import AbstractModelModule

__all__ = ("AbstractModelFunctions",)


class AbstractModelFunctions(AbstractModelModule):
    """Model functions container."""

    def _equals(self, other: object) -> bool:
        return super()._equals(other) and self.model.equals(other.model)
