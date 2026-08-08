from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression signals that the market is consolidating after a period of high volatility. "
        "During this phase, prices trade within a narrower range than usual. This consolidation often precedes a breakout, "
        "which can lead to higher returns when entered at the right moment."
    )

    def __init__(self, window: int = 20, compression_threshold: float = 1.5) -> None:
        self._window = window
        self._compression_threshold = compression_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {}
        for symbol in view.symbols:
            prices = [float(v) for v in history.select(["session_date", symbol]).to_dict(False)]
            high, low, close = max(p[1] for p in prices), min(p[1] for p in prices), prices[-1][1]
            range_ = (high - low) / close
            if range_ < self._compression_threshold:
                symbol_prices[symbol] = range_

        picks: list[str] = [s for s, r in symbol_prices.items() if r == min(symbol_prices.values())]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={p: weight for p in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest