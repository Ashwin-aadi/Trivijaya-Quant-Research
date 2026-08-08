from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySctf(Strategy):
    rationale = (
        "Trend following with a volatility-scaled threshold aims to capture the momentum of "
        "strong trends while adjusting for periods of high market volatility. This strategy "
        "is designed to be more conservative in volatile markets and aggressive when markets "
        "are calm."
    )

    def __init__(self, window: int = 20, threshold_multiplier: float = 1.5) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the 20-day log returns
        log_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).alias("r")
        history = history.with_columns(log_returns)
        
        # Calculate the rolling mean and standard deviation of returns
        rolling_mean = history.select(
            pl.col("r").rolling_mean(window_size=self._window, center=False).alias("mean_r")
        )
        rolling_std = history.select(
            pl.col("r").rolling_std(window_size=self._window, center=False).alias("std_r")
        )
        
        # Combine the mean and standard deviation
        combined_history = history.join(rolling_mean, on="session_date", how="left") \
                                  .join(rolling_std, on="session_date", how="left")
        
        # Calculate the threshold for entry
        threshold = combined_history["mean_r"] + self._threshold_multiplier * combined_history["std_r"]
        
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns or symbol not in combined_history.columns:
                continue
            
            latest_close = float(view.latest_close()[symbol])
            latest_mean_r = float(combined_history[combined_history["session_date"] == stamp]["mean_r"][0])
            latest_threshold = float(threshold[threshold["session_date"] == stamp]["std_r"][0]) + latest_mean_r
            if combined_history.select((pl.col("r") >= pl.lit(latest_threshold))).filter(pl.col("symbol") == symbol).height > 0:
                picks.append(symbol)
        
        picks = picks[:5]
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