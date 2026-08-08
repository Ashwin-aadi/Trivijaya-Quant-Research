from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion seeks to capitalize on the tendency of stocks that "
        "have deviated significantly from their historical average to return towards it. "
        "By identifying such deviations, we can exploit potential reversals."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        means = (
            history[symbols]
            .group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .collect()
        )

        closes = view.closes(lookback=self._window)
        deviations = [float(closes[col][-1] - means.get(col, "mean")) for col in symbols]
        
        if all(abs(d) < 0.05 for d in deviations):
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_candidates = [
            symbol
            for i, dev in enumerate(deviations)
            if abs(dev) > 0.1 and closes[symbols[i]][-1] / means[col]["mean"] < 0.95
        ]

        weight = 1.0 / len(mean_reversion_candidates) if mean_reversion_candidates else 0.0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in mean_reversion_candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest