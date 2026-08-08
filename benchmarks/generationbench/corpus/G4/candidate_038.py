from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to capitalize on the relationship between market volatility and trends in the Indian equity market. By scaling trading positions based on realized volatility, it seeks to capture trend profits while minimizing risk during volatile periods."
    )

    def __init__(self, window: int = 20, max_positions: int = 30) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        realized_volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("high") - pl.col("low")).rolling_sum(self._window) /
                (history["close"].shift(-1).to_list()[-self._window:])
            ).with_columns(
                (pl.col("high") / pl.col("low").shift(1) - 1.0).alias("daily_return"),
                ((pl.col("adj_close") - pl.col("adj_close").shift(1)) /
                 pl.col("adj_close").shift(1)).abs().rolling_sum(self._window).alias("realized_volatility")
            ).select(
                "symbol", "session_date", "realized_volatility"
            )
        )

        if realized_volatility.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank by volatility and select top N symbols
        ranked = realized_volatility.sort("realized_volatility").group_by("symbol").agg(
            pl.col("realized_volatility").mean().alias("avg_realized_volatility")
        ).sort("avg_realized_volatility", descending=True).head(self._max_positions)

        weights = {row["symbol"]: 1.0 / len(ranked) for _, row in ranked.iter_rows()}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest