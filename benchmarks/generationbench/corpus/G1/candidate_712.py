from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reverts to the mean over time. By identifying assets that are far from their "
        "trailing average and betting on a reversal, we aim to capture favorable price movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes(lookback=self._window).select(
            [pl.col(f"session_date.{date}")] + [pl.col(s) for s in view.symbols]
        )
        means = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("trailing_mean"))
            .join(latest_closes, on="symbol", how="inner")
        )

        symbols_with_data = [
            symbol
            for symbol in view.symbols
            if not means.filter(pl.col("symbol") == symbol).is_empty()
        ]

        if len(symbols_with_data) < self._window:
            return Signal(information_available_at=stamp, weights={})

        deviations = (
            history.join(means.select(["symbol", "trailing_mean"]), on="symbol")
            .with_columns(
                (pl.col("adj_close") - pl.col("trailing_mean")).alias("deviation")
            )
            .sort("deviation", descending=True)
            .head(self._window)
        )

        top_symbols = deviations.select("symbol").to_series().to_list()
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest