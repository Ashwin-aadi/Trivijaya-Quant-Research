from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression suggests that the market is in a consolidation phase, reducing "
        "volatility and preparing for potential breakout. Identifying stocks with reduced range"
        " can indicate a setup for future price action."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            high_values = [float(v) for v in history[symbol + "_high"].to_list()]
            low_values = [float(v) for v in history[symbol + "_low"].to_list()]
            range_widths = [h - l for h, l in zip(high_values, low_values)]
            avg_range = sum(range_widths) / len(range_widths)
            if all(r < 1.2 * avg_range for r in range_widths):
                picks.append(symbol)

        picks = picks[:5]
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