from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "Volatility-scaled trend following is a strategy that leverages the historical volatility "
        "of assets to identify and follow trends. High volatility periods suggest active price movement, "
        "potentially offering trading opportunities."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or len(history) < self._window:
            return Signal(information_available_at=stamp, weights={})

        daily_returns = (history["adj_close"] / history["adj_close"].shift(1) - 1.0).to_series()
        volatility = daily_returns.std()

        # Identify trending symbols based on their returns relative to the mean
        symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue

            recent_history = history.select(pl.col("adj_close").filter(pl.col("session_date") > date(2019, 12, 31)))
            recent_returns = (recent_history["adj_close"] / recent_history["adj_close"].shift(1) - 1.0).to_series()

            if not recent_returns.is_empty():
                trend_score = (recent_returns.mean() / volatility).abs()
                if trend_score >= self._threshold:
                    symbols.append(symbol)

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest