from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends while scaling position sizes based on volatility. "
        "During periods of high volatility, the strategy reduces exposure to protect capital."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility = (history["close"] / history["close"].shift(1) - 1.0).abs().mean()
        trend = history.sort("session_date").tail(20)["close"].last() / history.sort(
            "session_date"
        ).head(1)["close"].item()

        if volatility > self._threshold:
            weights = {s: 0.0 for s in view.symbols}
        else:
            weight = 1.0 / len(view.symbols)
            weights = {s: weight for s in view.symbols}

        return Signal(
            information_available_at=stamp,
            weights={k: float(v) for k, v in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest