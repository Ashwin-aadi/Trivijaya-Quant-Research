from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the persistence of long-term trends in equity markets while "
        "accounting for volatility. During periods of high market volatility, price movements tend "
        "to revert, whereas low-volatility periods often see sustained trending behavior. By scaling "
        "trend-following positions based on current volatility levels, higher-risk, lower-volatility "
        "periods result in smaller trades, reducing potential losses during market turbulence, while "
        "lower-risk, higher-volatility periods allow for larger positions to capture trending behavior."
    )

    def __init__(self, trend_window: int = 200, vol_window: int = 20, max_positions: int = 30) -> None:
        self._trend_window = trend_window
        self._vol_window = vol_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=1000)  # Ensure we have enough data to compute signals

        if history.height < (self._trend_window + self._vol_window):
            return Signal(information_available_at=stamp, weights={})

        trend_signal = (
            history.group_by("symbol")
            .agg(
                pl.col("close").rolling_mean(window_size=self._trend_window).alias(f"ma_{self._trend_window}"),
                pl.col("close").rolling_mean(window_size=200).alias(f"ma_200"),
            )
            .filter(pl.col(f"ma_{self._trend_window}") > pl.col(f"ma_200"))
        )

        vol_signal = history.group_by("symbol").agg(
            (pl.col("close") / pl.col("open") - 1).rolling_std(window_size=self._vol_window).alias("volatility")
        ).select(pl.col("symbol"), "volatility")

        merged = trend_signal.join(vol_signal, on="symbol", how="inner")
        ranked = (
            merged.sort(
                [pl.col(f"ma_{self._trend_window}").desc(), -pl.col("volatility")],
                descending=[True, False]
            ).head(self._max_positions)
            .with_columns(
                (1 / pl.col("volatility")).alias("weight"),
            )
        )

        if ranked.is_empty():
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(ranked["weight"].to_list())
        weights = {row["symbol"]: row["weight"] / total_weight for _, row in ranked.iter_rows()}

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest