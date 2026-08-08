from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion occurs when an asset's price moves towards its historical mean. "
        "Short-horizon mean reversion strategies exploit deviations from this mean by betting "
        "that the price will revert to a typical level."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        mean_close = (history.select(pl.col("adj_close").mean())  # type: ignore
                      .transpose()  # Make it long format for easier calculation
                      .with_columns((pl.col(0) - pl.col(1)).alias("deviation"))
                      .select("symbol", "deviation")
                      .collect())
        deviations = {row[0]: float(row[1]) for row in mean_close.to_dicts()}

        under_mean: list[str] = [s for s, d in deviations.items() if d < -1]
        over_mean: list[str] = [s for s, d in deviations.items() if d > 1]

        if not (under_mean or over_mean):
            return Signal(information_available_at=stamp, weights={})

        weight_under = 0.5 / len(under_mean)
        weight_over = -0.5 / len(over_mean)

        weights = {s: weight_under for s in under_mean}
        for s in over_mean:
            weights[s] += weight_over

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest