from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are a strategy based on the idea that significant "
        "trading volume often precedes strong price movements. By identifying symbols with both "
        "large price changes and corresponding high trading volumes, one can capture "
        "opportunities from these movements."
    )

    def __init__(self, lookback_days: int = 10) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and trading volume
        history = (
            history.select(
                pl.col("symbol"),
                "session_date",
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("return"),
                pl.col("volume").cast(pl.Float64),
            )
            .filter(pl.col("session_date") < view.as_of)
            .sort(by="session_date", descending=False)
        )

        # Calculate cumulative volume for each symbol
        history = (
            history.with_columns(
                (pl.col("volume").cumsum().over("symbol")).alias("cumulative_volume")
            )
            .group_by("symbol")
            .agg(pl.DataFrame(
                [
                    pl.sum("return").alias("total_return"),
                    pl.max("cumulative_volume").alias("max_cumulative_volume"),
                ]
            ))
        )

        # Select symbols with significant returns and high cumulative volume
        history = (
            history.filter(
                (pl.col("total_return") > 0.01)
                & (pl.col("max_cumulative_volume") >= 5 * pl.col("volume").mean())
            )
        ).sort(by="total_return", descending=True)

        # Get the top symbols
        top_symbols = [row["symbol"] for row in history.to_dicts()[:3]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight_per_symbol for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest