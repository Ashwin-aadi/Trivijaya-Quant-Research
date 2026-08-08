from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategies exploit the tendency of stocks in an upward "
        "trend to continue outperforming those in a downward trend. This is based on the "
        "idea that strong relative performance persists over time."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Compute returns for each stock
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol", maintaining_order=True)
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Get top and bottom performing stocks
        top_stocks = history_with_returns.sort("avg_return", descending=True).head(self._lookback // 4)["symbol"]
        bottom_stocks = history_with_returns.sort("avg_return").head(self._lookback // 4)["symbol"]

        # Calculate weights for the selected stocks
        weight_top = 0.6 / len(top_stocks)
        weight_bottom = -0.2 / len(bottom_stocks)

        top_weights = {s: weight_top for s in top_stocks}
        bottom_weights = {s: weight_bottom for s in bottom_stocks}

        weights = {**top_weights, **bottom_weights}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest