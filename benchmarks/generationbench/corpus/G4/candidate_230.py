from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits momentum during low-volatility periods by scaling position size "
        "based on recent volatility. It enters long positions in upward trends when the short-term "
        "moving average crosses above a longer-term trendline during calm market conditions."
    )

    def __init__(self, short_window: int = 50, long_window: int = 200, atr_window: int = 14, max_positions: int = 20) -> None:
        self._short_window = short_window
        self._long_window = long_window
        self._atr_window = atr_window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._short_window + 1, self._long_window + 1))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate moving averages
        sma_short = (
            history.select([pl.col("symbol"), pl.col("session_date"), (pl.col("close").rolling_mean(self._short_window)).alias(f"close_{self._short_window}")])
            .sort("session_date")
            .with_columns(pl.col(f"close_{self._short_window}").shift(-1).alias(f"close_{self._short_window}_shifted"))
        )
        
        sma_long = (
            history.select([pl.col("symbol"), pl.col("session_date"), (pl.col("close").rolling_mean(self._long_window)).alias(f"close_{self._long_window}")])
            .sort("session_date")
            .with_columns(pl.col(f"close_{self._long_window}").shift(-1).alias(f"close_{self._long_window}_shifted"))
        )
        
        # Calculate ATR
        high = history.select([pl.col("symbol"), pl.col("session_date"), "high"])
        low = history.select([pl.col("symbol"), pl.col("session_date"), "low"])
        close = history.select([pl.col("symbol"), pl.col("session_date"), "close"])
        
        true_range = (
            high.join(low, on=["symbol", "session_date"], how="inner")
                .join(close, on=["symbol", "session_date"], how="inner")
                .with_columns((pl.col("high") - pl.col("low")).alias("tr1"))
                .with_columns(((pl.col("high") - pl.col("close").shift(1)).abs()).alias("tr2"))
                .with_columns(((pl.col("low") - pl.col("close").shift(1)).abs()).alias("tr3"))
                .select([pl.all(), (pl.col("tr1").max().over("symbol").alias("true_range"))])
        )
        
        atr = (
            true_range.groupby("symbol")
                      .agg((pl.col("true_range").rolling_mean(self._atr_window).alias(f"atr_{self._atr_window}")))
                      .sort("session_date")
        )

        # Merge all data
        merged_data = sma_short.join(sma_long, on=["symbol", "session_date"], how="inner")
        merged_data = merged_data.join(atr, on=["symbol", "session_date"], how="inner")

        signals: list[str] = []
        for symbol in symbols:
            sma_short_val = float(merged_data.select(pl.col(f"close_{self._short_window}_shifted")).filter(pl.col("symbol") == symbol).last().to_list()[0])
            sma_long_val = float(merged_data.select(pl.col(f"close_{self._long_window}_shifted")).filter(pl.col("symbol") == symbol).last().to_list()[0])
            atr_val = float(atr.filter(pl.col("symbol") == symbol).select(pl.col(f"atr_{self._atr_window}")).last().to_list()[0])

            if sma_short_val > sma_long_val and atr_val < (merged_data.select(pl.col(f"atr_{self._atr_window}").rolling_mean(21)).filter(pl.col("symbol") == symbol).last().to_list()[0]):
                signals.append(symbol)

        weights = {s: 1.0 / len(signals) for s in signals}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest