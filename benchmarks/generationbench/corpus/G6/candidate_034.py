from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalIntradayTrading(Strategy):
    rationale = (
        "This strategy capitalizes on historical seasonal patterns in the Indian market, "
        "particularly focusing on months with significant economic events such as month-end earnings announcements."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_data = (
            history.select(["symbol", "session_date", pl.col("high").alias("H"), pl.col("low").alias("L")])
            .sort("session_date", descending=True)
            .head(self._window)
        )

        high_above_ma = 1.01
        low_above_ma = 1.01

        symbols_with_highs_above_ma = (
            recent_data.groupby("symbol")
            .agg(
                (pl.col("H") > pl.col("close").shift(-self._window) * high_above_ma).alias("high_above_ma"),
                (pl.col("L") > pl.col("close").shift(-self._window) * low_above_ma).alias("low_above_ma"),
            )
            .filter((pl.col("high_above_ma") == True) | (pl.col("low_above_ma") == True))
        )

        eligible_symbols = symbols_with_highs_above_ma["symbol"].to_list()[: self._top_n]

        if not eligible_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(eligible_symbols)
        weights = {symbol: weight_per_symbol for symbol in eligible_symbols}

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest