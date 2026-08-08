from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when price volatility decreases, suggesting that the market "
        "is becoming less certain about future prices. This can lead to mean reversion, where "
        "prices tend to move back towards their historical average range."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        compressed_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            opens = [float(o) for o in history[symbol].select("open").to_list()[0]]
            closes = [float(c) for c in history[symbol].select("close").to_list()[0]]
            high_low_spread = max(opens) - min(closes)
            mean_range = (max(opens) - min(closes)) / self._window
            if high_low_spread < 1.5 * mean_range:
                compressed_symbols.append(symbol)

        compressed_symbols = list(set(compressed_symbols))
        if not compressed_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(compressed_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in compressed_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest