from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion is a phenomenon where asset prices that have deviated significantly "
        "from their historical mean tend to revert back. In the short term, stocks that have "
        "underperformed compared to their peers are more likely to outperform in the near future."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Compute the mean close price for each symbol over the lookback period
        symbol_means = (
            closes.group_by("symbol")
                  .agg(pl.col("adj_close").mean().alias("mean"))
                  .with_columns((pl.col("adj_close") - pl.col("mean")).abs().alias("deviation_from_mean"))
        )

        # Rank symbols by their deviation from the mean
        ranked = symbol_means.sort("deviation_from_mean", descending=True)

        picks: list[str] = []
        for i in range(len(view.symbols)):
            symbol = ranked["symbol"].to_list()[i]
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            if values[-1] <= min(values):
                picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest