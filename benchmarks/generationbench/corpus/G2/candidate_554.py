from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by scaling the position size "
        "by the recent volatility of an asset. High volatility assets may be expected to have "
        "large price movements, and thus larger positions can be taken with a similar risk "
        "profile compared to low-volatility assets."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = (
            (history["close"].shift(-1) - history["close"]).abs()
            / history["volume"]
        ).mean().item()

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            recent_closes = [float(v) for v in history[symbol].to_list()]
            if len(recent_closes) < self._window:
                continue
            trend = (
                (recent_closes[-1] - recent_closes[0])
                / sum([abs(c2 - c1) for c1, c2 in zip(recent_closes[:-1], recent_closes[1:])])
            )
            if abs(trend) < 0.1:  # avoid zero division and small trends
                continue

            weights[symbol] = volatility * trend

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest