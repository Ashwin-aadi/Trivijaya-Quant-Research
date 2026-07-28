"""Category ``point_in_time_bypass``, variant ``obvious``.

Defect: the strategy keeps the price frame it was constructed with and reads it during
``generate``. The ``MarketView`` it is handed has been truncated to sessions strictly before the
decision date; the stored frame has not. Nothing here indexes forward on purpose — the filter does
not mention a date at all — but that is exactly the problem: with no date restriction the series
runs to the end of the sample, so the mean and the standard deviation below are computed over the
whole thing, future included.

The category is about the route rather than the arithmetic. Whatever the strategy computes after
reaching around the view is computed on data it was not supposed to be able to reach.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from src.backtest.strategy import MarketView, Signal, Strategy


class PointInTimeBypassObvious(Strategy):
    """Ranks names on a z-score computed from the stored price frame."""

    rationale = (
        "Comparing a name's current price with its own recent distribution is a cleaner reversion "
        "signal than comparing it with the cross-section, because it does not assume the names "
        "are comparable to each other. Standardising by the name's own dispersion means the "
        "threshold has the same meaning for a quiet counter and a violent one."
    )

    def __init__(self, panel: pl.DataFrame, top_n: int = 10) -> None:
        self._panel = panel
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _stamp(view)
        latest = view.latest_close()
        if not latest:
            return Signal(information_available_at=stamp, weights={})

        scores: dict[str, float] = {}
        for symbol, price in latest.items():
            # THE CHEAT: the stored frame is read here, inside the decision function. The `view`
            # argument is the only object in scope that has been truncated to the decision moment;
            # this one still holds every session in the sample.
            series = self._panel.filter(pl.col("symbol") == symbol)["adj_close"]
            if series.len() < 2:
                continue
            spread = _as_float(series.std())
            if spread <= 0:
                continue
            scores[symbol] = (price - _as_float(series.mean())) / spread

        chosen = sorted(scores, key=lambda s: (scores[s], s))[: self._top_n]
        return Signal(information_available_at=stamp, weights=_spread(chosen))


def _as_float(value: object) -> float:
    """Coerce a polars aggregate to a float; anything unusable becomes zero."""
    return float(value) if isinstance(value, int | float) else 0.0


def _stamp(view: MarketView) -> date:
    """Latest session the strategy is entitled to have seen."""
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _spread(names: list[str]) -> dict[str, float]:
    """Equal weights across ``names``; empty in, empty out."""
    if not names:
        return {}
    return dict.fromkeys(names, 1.0 / len(names))
