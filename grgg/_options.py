from copy import deepcopy
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self


@dataclass(slots=True)
class Options:
    __original__: Self = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.__original__ = deepcopy(self)

    def __copy__(self) -> Self:
        return deepcopy(self)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._restore()

    def _restore(self) -> None:
        if self.__original__ is not None:
            for slot_name in self.__slots__:
                obj = getattr(self.__original__, slot_name)
                setattr(self, slot_name, deepcopy(obj))


@dataclass(slots=True)
class KernelOptions(Options):
    logspace: bool = True
    eps: float = 1e-6


@dataclass(slots=True)
class SampleOptions(Options):
    batch_size: int = 1000


@dataclass(slots=True)
class OptimizeMuOptions(Options):
    beta_max: float = 1e2


@dataclass(slots=True)
class OptimizeOptions(Options):
    method: str = "Nelder-Mead"
    tol: float = 1e-3
    mu: OptimizeMuOptions = field(default_factory=OptimizeMuOptions)


@dataclass(slots=True)
class TopOptions(Options):
    kernel: KernelOptions = field(default_factory=KernelOptions)
    sample: SampleOptions = field(default_factory=SampleOptions)
    optimize: OptimizeOptions = field(default_factory=OptimizeOptions)


options = TopOptions()
