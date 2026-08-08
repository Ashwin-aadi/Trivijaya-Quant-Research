from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion(Strategy):
    rationale = (
        "Price reversion strategies capitalize on the tendency of asset prices to revert "
        "to recent mean levels. By identifying deviations from a trailing moving average, "
        "we can generate buy and sell signals."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = (closes.mean().to_series()).item()
        deviations = [
            abs(float(c) - mean_close) / mean_close
            for c in closes["close"].drop_nulls().to_list()
        ]
        
        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            deviation = deviations[-1]
            if deviation > self._threshold:
                picks.append(symbol)

        picks = picks[:5]  # Top N symbols with highest deviation
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