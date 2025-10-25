from typing import ClassVar

from grgg.statistics.abc import VT, AbstractErgmNodeStatistic


class Degree(AbstractErgmNodeStatistic):
    module: VT

    label: ClassVar[str] = "degree"
    supported_moments: ClassVar[tuple[int, ...]] = (1,)
    supports_monte_carlo: ClassVar[bool] = False
