from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price action becomes more volatile over time. "
        "This can indicate potential reversals or continuation patterns. By identifying stocks with "
        "recent range compression, we may be able to capture profitable trades."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            open_prices = [float(o) for o in history[symbol + "_open"].to_list()]
            close_prices = [float(c) for c in history[symbol + "_close"].to_list()]

            if len(open_prices) < self._window or len(close_prices) < self._window:
                continue

            high_low_range = max([h - l for h, l in zip(history[symbol + "_high"].to_list(), history[symbol + "_low"].to_list())])
            recent_high_low_range = max([h - l for h, l in zip(close_prices[:-1], close_prices[1:])]) if len(close_prices) > 2 else 0
            range_ratio = high_low_range / recent_high_low_range

            if range_ratio > 3.0:
                picks.append(symbol)

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