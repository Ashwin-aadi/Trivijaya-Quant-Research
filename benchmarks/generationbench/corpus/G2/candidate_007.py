from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the price of a stock oscillates within a narrower range "
        "than its historical average. This phenomenon is often associated with increased liquidity and "
        "greater trading interest, which can lead to higher volatility and potentially profitable "
        "trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
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
            high_low_diffs = (
                history[f"{symbol}_high"] - history[f"{symbol}_low"]
            ).to_list()
            mean_range = sum(high_low_diffs) / len(high_low_diffs)
            recent_range = high_low_diffs[-1]
            if recent_range < 0.9 * mean_range:
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