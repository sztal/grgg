from plum import Dispatcher

__all__ = ("dispatch",)


dispatch = Dispatcher(warn_redefinition=True)
