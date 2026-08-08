from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion seeks to exploit the tendency of stock prices to revert to their mean. "
        "If a stock has significantly deviated from its historical average price, it is expected "
        "to move back towards that average, providing an opportunity for profit."
    )

    def __init__(self, window: int = 10, threshold: float = 2.0) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_close = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("m"))
            .with_columns((pl.col("adj_close") - pl.col("m")).abs().alias("deviation"))
        )

        for symbol in view.symbols:
            if symbol not in mean_close["symbol"].to_list():
                continue

            current_close = history.filter(pl.col("session_date") == stamp).get_column(
                "adj_close"
            )[0]
            mean_close_value = mean_close.filter(pl.col("symbol") == symbol)["m"][0]

            deviation = abs(current_close - mean_close_value)
            if deviation > self._threshold * (mean_close_value / 100):
                return Signal(
                    information_available_at=stamp,
                    weights={symbol: 1.0},
                )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest