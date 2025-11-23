from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from dataclasses import replace
from types import TracebackType
from typing import Any, Self

from pydantic import NonNegativeFloat, PositiveInt
from pydantic.dataclasses import Field, dataclass
from rich.pretty import pprint


@dataclass(slots=True)
class Options(MutableMapping[str, Any]):
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
        self.reset()

    def __setattr__(self, name: str, value: Any) -> None:
        new = self.replace(**{name: value})
        object.__setattr__(self, name, new[name])

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def __delitem__(self, name: str) -> None:
        errmsg = "options cannot be deleted"
        raise TypeError(errmsg)

    def __iter__(self) -> Iterator[str]:
        return iter(self.__slots__)

    def __len__(self) -> int:
        return len(self.__slots__)

    def reset(self) -> None:
        """Reset options to their original values."""
        new = self.__class__()
        for slot_name in self.__slots__:
            obj = getattr(new, slot_name)
            setattr(self, slot_name, obj)

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the options with specified changes."""
        return replace(self, **changes)

    def show(self) -> None:
        """Pretty-print the options."""
        pprint(self)

    def get(self, option: str, value: Any = None) -> Any:
        """Get an option value with a default."""
        if option not in self.__dataclass_fields__:
            errmsg = f"option '{option}' does not exist"
            raise AttributeError(errmsg)
        if value is None:
            return getattr(self, option)
        return getattr(self.replace(**{option: value}), option)

    def set(self, option: str, value: Any) -> None:
        """Set an option value."""
        new = self.replace(**{option: value})
        setattr(self, option, getattr(new, option))


@dataclass(slots=True)
class RandomGraphOptions(Options):
    mu: float = 0.0


@dataclass(slots=True)
class GeometricOptions(Options):
    beta: NonNegativeFloat = 1.5
    log: bool = True
    eps: NonNegativeFloat = 1e-9


@dataclass(slots=True)
class ModelOptions(Options):
    random_graph: RandomGraphOptions = Field(default_factory=RandomGraphOptions)
    geometric: GeometricOptions = Field(default_factory=GeometricOptions)


@dataclass(slots=True)
class SamplingOptions(Options):
    batch_size: PositiveInt = 10000


@dataclass(slots=True)
class LoopOptions(Options):
    batch_size: PositiveInt | None = 1000
    unroll: PositiveInt = 2


@dataclass(slots=True)
class MonteCarloOptions(Options):
    mc: PositiveInt | bool = 50
    repeat: PositiveInt = 1
    average: bool = True
    same_seed: bool = True

    def __post_init__(self) -> None:
        if self.mc is True:
            self.mc = options.monte_carlo.mc

    @classmethod
    def from_size(cls, n: int, mc: int | None = None, **kwargs: Any) -> Self:
        """Create Monte Carlo options from model size."""
        if mc is None:
            mc = options.monte_carlo.mc if n > options.auto.mc else False
        opts = cls(mc=mc, **kwargs)
        return opts


@dataclass(slots=True)
class ProgressOptions(Options):
    description: str = "Processing..."
    disable: bool = False

    @classmethod
    def from_steps(
        cls, steps: float, progress: bool | Mapping | None = None, **kwargs: Any
    ) -> Self:
        """Create progress options from number of steps."""
        if isinstance(progress, bool):
            disable = not progress
            opts = kwargs
        elif progress is None:
            disable = steps <= options.auto.progress
            opts = kwargs
        else:
            disable = progress.pop("disable", False)
            opts = {**progress, **kwargs}
        return cls(disable=disable, **opts)


@dataclass(slots=True)
class AutoOptions(Options):
    mc: PositiveInt = 1000
    progress: NonNegativeFloat = 1.0


@dataclass(slots=True)
class PackageOptions(Options):
    model: ModelOptions = Field(default_factory=ModelOptions)
    sampling: SamplingOptions = Field(default_factory=SamplingOptions)
    loop: LoopOptions = Field(default_factory=LoopOptions)
    monte_carlo: MonteCarloOptions = Field(default_factory=MonteCarloOptions)
    progress: ProgressOptions = Field(default_factory=ProgressOptions)
    auto: AutoOptions = Field(default_factory=AutoOptions)


options = PackageOptions()
