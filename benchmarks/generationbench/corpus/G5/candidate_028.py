from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Reversion to the mean is a statistical phenomenon where prices that deviate from "
        "their historical average are likely to move back towards it. This strategy looks at "
        "each stock's price relative to its trailing 20-day mean and enters trades when the "
        "price falls below this level or exceeds it significantly, balancing long and short positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_df = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("trailing_mean"))
        )
        
        closes_wide = view.closes(lookback=self._window + 1)

        merged = closes_wide.join(mean_df, on="symbol", how="inner")

        if merged.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        long_signals: list[str] = []
        short_signals: list[str] = []

        for symbol in view.symbols:
            if symbol not in merged.columns or symbol not in history.columns:
                continue

            close_values = [float(v) for v in merged[symbol].to_list()]
            mean_value = float(merged[f"trailing_mean_{symbol}"].first())
            
            if len(close_values) < self._window:
                continue
            
            last_close = close_values[-1]
            mean_last_close_ratio = last_close / mean_value
            # Consider positions based on the ratio of last close to trailing mean
            if 0.95 <= mean_last_close_ratio <= 1.05:  # Within a reasonable range around the mean
                continue
            
            if last_close < mean_value:
                long_signals.append(symbol)
            elif last_close > mean_value * 1.5:  # Aggressive short signal
                short_signals.append(symbol)

        combined_signals = long_signals + short_signals
        if not combined_signals:
            return Signal(information_available_at=stamp, weights={})

        weight_per_signal = 1.0 / len(combined_signals)
        return Signal(
            information_available_at=stamp,
            weights={
                s: weight_per_signal for s in combined_signals
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest