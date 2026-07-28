"""Category ``snooped_parameter`` (form 9a), variant ``reworded``.

Defect: identical to ``snooped_parameter_obvious``. A sweep over candidate span values scores each
one against the whole supplied frame and retains the highest scorer, so the span the strategy runs
with was selected against the data it will be judged on, and the rejected candidates are never
counted as trials.

No identifier in the code body describes what is happening. There is no mention of optimisation,
tuning, selection, or a best value. The structure is intact and is what a detector should be
looking at: a fixed collection of candidate settings, a scoring call applied to each, and an
extremum over the results assigned to an instance attribute that the trading rule then depends on.
Any of ``max``, ``min``, ``sorted(...)[0]`` or an explicit running comparison spells the same thing.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy

_SPANS = (21, 42, 63, 126, 252)


class WindowSweepRanker(Strategy):
    """Ranks on a trailing change measured over the retained span."""

    rationale = (
        "Momentum needs a horizon and there is no theory that fixes it, so the horizon is chosen "
        "empirically from a small set of standard candidates rather than asserted. Restricting "
        "the set to five well-known windows keeps the search small enough that the chosen value "
        "is not simply the best of a hundred coin flips."
    )

    def __init__(self, reference_set: pl.DataFrame, bucket: int = 10) -> None:
        self._bucket = bucket
        table = {span: _rate(reference_set, span) for span in _SPANS}
        self._span = max(table, key=lambda s: table[s])

    def generate(self, view: MarketView) -> Signal:
        marker = _marker(view)
        closes = view.closes(lookback=self._span + 1)
        if closes.height < self._span:
            return Signal(information_available_at=marker, weights={})

        table: dict[str, float] = {}
        for name in view.symbols:
            if name not in closes.columns:
                continue
            values = [float(v) for v in closes[name].drop_nulls().to_list()]
            if len(values) < self._span or values[0] <= 0:
                continue
            table[name] = values[-1] / values[0] - 1.0
        picks = sorted(table, key=lambda s: (-table[s], s))[: self._bucket]
        return Signal(information_available_at=marker, weights=_allocate(picks))


def _rate(rows: pl.DataFrame, span: int) -> float:
    """Spread between the upper and lower deciles of the trailing change at this span."""
    measured = (
        rows.sort(["symbol", "session_date"])
        .with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(span).over("symbol") - 1.0).alias(
                "delta_b"
            )
        )
        .drop_nulls("delta_b")
    )
    if measured.is_empty():
        return 0.0
    upper = measured["delta_b"].quantile(0.9) or 0.0
    lower = measured["delta_b"].quantile(0.1) or 0.0
    return float(upper) - float(lower)


def _marker(view: MarketView) -> date:
    """Most recent session in the visible window."""
    seen = view.history()
    if seen.is_empty():
        return date(1900, 1, 1)
    newest = seen["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _allocate(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
