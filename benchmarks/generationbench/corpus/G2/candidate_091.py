from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when volatility decreases, causing prices to move more "
        "tightly within a narrow range. This can lead to mean reversion, where price levels that "
        "have been compressed are likely to revert back to their historical average. By identifying "
        "assets with high dispersion in recent returns, we aim to capture the potential for mean reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the percentage change in adj_close for each symbol
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
        )

        # Compute the mean return and standard deviation of returns over the window
        stats = (
            history.group_by("symbol")
            .agg(
                {
                    "daily_return": [
                        (pl.col("daily_return").mean().alias("mean_return")),
                        (pl.col("daily_return").std().alias("std_return")),
                    ]
                }
            )
            .collect()
        )

        # Filter out symbols with insufficient data
        stats = stats.filter(pl.col("daily_return.length") >= self._window)

        # Sort by standard deviation to find symbols with the highest volatility
        sorted_stats = stats.sort("std_return", descending=True).head(10)
        top_symbols = [row["symbol"] for row in sorted_stats.rows() if "symbol" in row]

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