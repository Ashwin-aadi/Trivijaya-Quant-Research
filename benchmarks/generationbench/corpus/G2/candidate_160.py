from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price levels revert to their mean over time. By identifying assets that have deviated "
        "significantly from their historical price range and betting on a reversion, we can capture "
        "momentum when prices return to more typical levels."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("mean")))
            .with_columns((pl.col("adj_close") - pl.col("mean")).alias("deviation"))
        )

        symbols = view.symbols
        signal_weights: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in mean_close["symbol"].to_list():
                continue

            mean_deviation = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date")
                .select(
                    (pl.col("adj_close").mean() - pl.col("close").mean()).alias("deviation"),
                    pl.col("symbol").first().alias("symbol"),
                )
            ).row(0)[0]

            recent_close = view.latest_close()[symbol]
            deviation_from_mean = abs(recent_close - mean_deviation)

            if deviation_from_mean > 1.5 * mean_deviation:
                signal_weights[symbol] = 1.0 / len(symbols)

        return Signal(information_available_at=stamp, weights=signal_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest