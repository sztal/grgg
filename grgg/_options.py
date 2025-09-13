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
        if self.__original__ is not None:
            for slot_name in self.__slots__:
                obj = getattr(self.__original__, slot_name)
                setattr(self, slot_name, deepcopy(obj))


@dataclass(slots=True)
class LayerOptions(Options):
    beta: float = 1.5
    mu: float = 0.0
    log: bool = True
    eps: float = 1e-9


@dataclass(slots=True)
class BatchOptions(Options):
    size: int = 1000
    auto_progress: int = 5000


@dataclass(slots=True)
class IntegrateOptions(Options):
    batch_size: int = 1000


@dataclass(slots=True)
class QuantizeOptions(Options):
    auto: bool = True
    n_bins: int = 256
    strategy: str = "joint"


@dataclass(slots=True)
class OptimizeMuOptions(Options):
    beta_max: float = 1e2


@dataclass(slots=True)
class OptimizeOptions(Options):
    method: str = "Nelder-Mead"
    tol: float = 1e-2
    mu: OptimizeMuOptions = field(default_factory=OptimizeMuOptions)


@dataclass(slots=True)
class TopOptions(Options):
    layer: LayerOptions = field(default_factory=LayerOptions)
    batch: BatchOptions = field(default_factory=BatchOptions)
    integrate: IntegrateOptions = field(default_factory=IntegrateOptions)
    quantize: QuantizeOptions = field(default_factory=QuantizeOptions)
    optimize: OptimizeOptions = field(default_factory=OptimizeOptions)


options = TopOptions()
