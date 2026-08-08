from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the tendency of assets to continue trending "
        "in their current direction after a period of high volatility. High volatility periods "
        "are typically followed by mean-reverting behavior, making it profitable to hold trends "
        "during such times."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_scaled_signals: dict[str, float] = {}
        for symbol in view.symbols:
            recent_closes = [float(v) for v in history["adj_close"][symbol].drop_nulls().to_list()]
            if len(recent_closes) < self._window:
                continue

            # Calculate returns
            returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1]
                       for i in range(1, len(recent_closes))]

            # Calculate volatility
            volatility = pl.DataFrame({"returns": returns}).select(
                (pl.col("returns").abs().mean() * 252).alias("volatility")
            ).collect()[0][0]

            # Trend direction
            trend_direction = max(returns) if recent_closes[-1] > recent_closes[0] else min(returns)

            # Volatility-scaled signal
            volatility_scaled_signal = abs(trend_direction / volatility)
            volatility_scaled_signals[symbol] = volatility_scaled_signal

        # Filter and select top symbols based on the scaled signals
        picks = [symbol for symbol, signal in sorted(volatility_scaled_signals.items(), key=lambda item: -item[1])][:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest