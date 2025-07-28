from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib as mpl
import seaborn as sns


def theme(
    params: Mapping[str, Any] | None = None,
    *,
    markers: Iterable[str] = ("o", "s", "X", "D", "^", "v"),
    colors: Iterable | None = None,
) -> dict[str, Any]:
    """
    Return a dictionary of matplotlib parameters for consistent styling.
    """
    markers = list(markers)
    if not colors:
        colors = sns.diverging_palette(
            0, 250, s=200, l=30, center="light", n=len(markers)
        )
    else:
        colors = list(colors)
    params = params or {}
    return {
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.labelsize": "x-large",
        "figure.titlesize": "xx-large",
        "axes.prop_cycle": mpl.cycler(marker=markers, color=colors),
        **params,
    }
