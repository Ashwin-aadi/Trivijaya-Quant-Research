from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to identify and ride strong trends while "
        "adjusting the investment size based on recent volatility. High volatility periods "
        "are expected to have more noise, so smaller positions are taken."
    )

    def __init__(self, window: int = 20, vol_window: int = 10, max_position: float = 0.5) -> None:
        self._window = window
        self._vol_window = vol_window
        self._max_position = max_position

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in symbols:
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue

            # Calculate returns
            returns = [(prices[i] / prices[i-1] - 1.0) for i in range(1, len(prices))]

            # Calculate volatility
            vol = pl.DataFrame({"return": returns}).select(
                (pl.col("return").abs().rolling_sum(self._vol_window) / self._vol_window).alias("vol")
            ).height

            if vol == 0:
                continue

            position_size = min(self._max_position, abs(sum(returns)))
            position_size /= vol
            position_size = max(-self._max_position, position_size)

            signals[symbol] = position_size if position_size > 0 else 0.0

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest