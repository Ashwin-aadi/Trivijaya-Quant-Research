from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion200d(Strategy):
    rationale = (
        "This strategy capitalizes on mean reversion in stock prices relative to their historical "
        "price levels. Asset prices tend to move towards their long-term average over time due to "
        "market inefficiencies and investor psychology."
    )

    def __init__(self, lookback: int = 200, threshold_std_dev: float = 1) -> None:
        self._lookback = lookback
        self._threshold_std_dev = threshold_std_dev

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._lookback).select("session_date", *view.symbols)
        adj_closes = closes["adj_close"].to_list()
        symbols = view.symbols

        # Calculate the 200-day simple moving average
        sma = history.group_by("symbol").agg(
            (pl.col("adj_close").mean().alias("sma"))
        ).collect()

        # Join with current closes to get the difference and standard deviation
        merged = sma.join(closes, on="symbol", how="inner")
        merged = merged.with_columns(
            (pl.col("adj_close") - pl.col("sma")).alias("diff"),
            ((pl.col("adj_close").std().over("symbol")) * self._threshold_std_dev).alias("std_diff"),
        )

        # Identify stocks that are significantly deviating from the mean
        filtered = merged.filter(
            (pl.col("diff") < -pl.col("std_diff"))
            | (pl.col("diff") > pl.col("std_diff"))
        )

        top_n_symbols: list[str] = []
        for symbol in symbols:
            if symbol not in filtered.columns:
                continue
            diff_val = float(filtered.filter(pl.col("symbol") == symbol)["diff"].item())
            std_dev_val = float(filtered.filter(pl.col("symbol") == symbol)["std_diff"].item())

            if abs(diff_val) > std_dev_val:
                top_n_symbols.append(symbol)

        top_n_symbols = top_n_symbols[:20]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest