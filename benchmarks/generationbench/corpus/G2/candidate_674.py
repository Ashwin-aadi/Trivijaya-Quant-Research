from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by adjusting the size of "
        "positions based on recent volatility. High volatility periods suggest that a strong "
        "trend is likely, while low volatility indicates a period of consolidation or range "
        "bound trading. By scaling positions according to historical volatility, the strategy "
        "can potentially benefit from trending markets without being overly exposed during "
        "periods of mean reversion."
    )

    def __init__(self, window: int = 20, scale_factor: float = 1.5) -> None:
        self._window = window
        self._scale_factor = scale_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        price_changes = (
            (history["adj_close"].to_list()[1:] / history["adj_close"].to_list()[:-1] - 1.0).alias("daily_return")
        )
        history = history.with_columns(price_changes)

        # Compute rolling mean and standard deviation of returns for volatility scaling
        mean_returns = pl.DataFrame(history.sort("session_date").tail(self._window)["daily_return"]).with_columns(
            (pl.col("daily_return").mean().over([0]).alias("mean_return"))
        )
        std_dev_returns = (
            history.select(pl.col("daily_return")).rolling_window(2, 1).std().sort("session_date").tail(self._window)
        ).with_columns((pl.col("daily_return_std").shift(-1)).fill_null(0.0).alias("std_dev_return"))

        combined = mean_returns.join(std_dev_returns, on="session_date")
        recent_close = view.latest_close()
        
        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in combined.columns or symbol not in recent_close.keys():
                continue
            daily_return = recent_close[symbol] - history[history["symbol"] == symbol]["adj_close"].item() / 100.0
            std_dev = float(combined[combined["symbol"] == symbol]["std_dev_return"])
            mean_return = float(combined[combined["symbol"] == symbol]["mean_return"])

            # Scale position based on volatility
            scaled_factor = self._scale_factor * (daily_return - mean_return) / std_dev
            if scaled_factor > 0:
                signals[symbol] = min(scaled_factor, 1.0)

        return Signal(information_available_at=stamp, weights={s: w for s, w in signals.items() if w > 0})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest