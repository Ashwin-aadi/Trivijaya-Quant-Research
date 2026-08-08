from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrend(Strategy):
    rationale = (
        "Volatility-scaled trend following seeks to capture trends by scaling positions "
        "based on historical volatility. High volatility periods suggest increased risk, "
        "thus reducing exposure; low volatility periods indicate reduced risk, thus increasing exposure."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        volatilities = (
            history.select(daily_returns)
                   .select(pl.col("r").rolling_std(window=self._window))
                   .with_columns((pl.col("r") > pl.col("r").shift(1)).alias("up"))
                   .filter(pl.col("up"))
                   .select(pl.col("r").mean().alias("avg_ret"))
        )

        if volatilities.is_empty():
            return Signal(information_available_at=stamp, weights={})

        average_return = float(volatilities.select("avg_ret")[0][0])
        symbol_volatilities = (
            history
                   .select(pl.col("symbol"), daily_returns)
                   .group_by("symbol")
                   .agg((pl.col("r").rolling_std(window=self._window)).alias("volatility"))
        )

        weights: dict[str, float] = {}
        for _, row in symbol_volatilities.iter_rows():
            volatility = float(row["volatility"])
            weight = 1.0 / (self._scale_factor * volatility) if volatility > 0 else 0.0
            if weight > 0:
                weights[row["symbol"]] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().to_list()[0]
    assert isinstance(newest, date)
    return newest