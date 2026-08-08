from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can lead to "
        "sustained price movements. This strategy identifies such moves by combining direction with"
        "volume signals."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_close = view.latest_close()

        # Calculate daily returns and volume for each symbol
        returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        volume = pl.col("volume").alias("v")

        df_returns = (
            history.with_columns([returns, volume])
                   .sort("session_date", descending=False)
                   .group_by("symbol")
                   .agg([
                       (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("avg_ret"),
                       pl.col("volume").sum().alias("total_volume")
                   ])
        )

        # Filter out symbols with insufficient data
        df_returns = df_returns.filter(pl.col("avg_ret").is_not_null() & (pl.col("total_volume") > 0))

        if df_returns.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = df_returns.sort("avg_ret", descending=True).head(self._top_n)
        symbols = [row["symbol"] for row in top_symbols.to_dicts()]

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().to_list()[0]
    assert isinstance(newest, date)
    return newest