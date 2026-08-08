from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy aims to tilt the portfolio towards low-volatility stocks to manage risk effectively while "
        "leveraging the diversified benefits of a basket of such stocks. By selecting the bottom 30% based on 20-day "
        "realized volatility and rebalancing regularly, we aim to achieve consistent performance with controlled risk."
    )

    def __init__(self, window: int = 20, threshold_percentile: float = 0.3, min_volume: int = 5_000_000) -> None:
        self._window = window
        self._threshold_percentile = threshold_percentile
        self._min_volume = min_volume

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        realized_volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close").shift(-self._window) / pl.col("adj_close") - 1.0).alias("returns")
            )
            .with_columns((pl.col("returns").rolling_std(window=self._window, min_periods=1)).alias("volatility"))
            .group_by("symbol")
            .agg(pl.col("volatility").mean().alias("mean_volatility"))
        )

        if realized_volatility.height < 2:
            return Signal(information_available_at=stamp, weights={})

        low_vol_symbols = (
            realized_volatility.sort("mean_volatility", descending=False)
            .with_column((pl.arange(1, pl.col("shape")[0] + 1) / pl.shape[0]).alias("rank"))
            .filter(pl.col("rank") <= self._threshold_percentile)
        ).select("symbol").to_series().to_list()

        liquidity = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("volume").sum()).alias("total_volume"),
                (pl.col("adj_close") == pl.col("adj_close")).count().alias("non_null_count")
            )
            .filter(pl.col("total_volume") >= self._min_volume and pl.col("non_null_count") == history.height)
        ).select("symbol").to_series().to_list()

        selected_symbols = set(low_vol_symbols).intersection(set(liquidity))

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest