from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum involves buying stocks that have outperformed their peers "
        "in the recent past. This strategy leverages the idea that stocks with strong relative performance are more likely to continue performing well."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=True)
            .select(["symbol", "session_date", "r"])
        )

        # Compute cumulative returns over the lookback period
        history = (
            history.group_by("symbol")
            .agg(
                pl.col("r").sum().alias("total_return"),
                pl.col("adj_close").last().alias("latest_close"),
            )
            .sort("total_return", descending=True)
        )

        symbols = [row["symbol"] for row in history.to_dicts()][: self._lookback]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        raise ValueError("No historical data available to generate signal.")
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest