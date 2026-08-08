from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility periods often precede reversals in trend. By focusing on "
        "volatility-scaled trends, we aim to identify and capitalize on potential direction "
        "changes in the market."
    )

    def __init__(self, window: int = 20, vol_window: int = 10) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + self._vol_window - 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate log returns
        returns = (closes["adj_close"] / closes["adj_close"].shift(1) - 1.0).alias("r")
        returns_df = closes.with_columns(returns)

        # Volatility calculation using rolling standard deviation
        volatility = (
            returns_df.select(
                pl.col("r").rolling_std(window_size=self._vol_window, center=False)
            )
            .with_column(pl.lit(self._window).alias("window"))
            .select(pl.col("r.rolling_std").alias("volatility"))
        )

        # Calculate trend by comparing current close to the mean of the past window
        trend = (
            returns_df.join(volatility, on="session_date")
            .with_columns(
                (pl.col("adj_close") - pl.col("adj_close").shift(self._window)).alias(
                    "trend"
                )
            )
            .select(
                pl.when(pl.col("volatility") > 0)
                .then((pl.col("trend") / pl.col("volatility")) * 1.5 + 1)
                .otherwise(0.0)
                .alias("trend_score")
            )
        )

        # Identify symbols with high trend scores
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in trend.columns:
                continue
            values = [float(v) for v in trend[symbol].to_list()]
            score = max(values)
            if score > 0.5:
                picks.append(symbol)

        picks = picks[:10]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest