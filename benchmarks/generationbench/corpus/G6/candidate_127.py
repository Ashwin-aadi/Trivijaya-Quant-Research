from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Dispersion20d(Strategy):
    rationale = (
        "This strategy selects stocks based on their position within the top or bottom deciles "
        "of daily high-low range over a 20-day window for entry. Exit occurs when the ATR falls "
        "below 1.5 standard deviations from the mean over a short-term period (e.g., 5 days)."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        high_low_diffs = [(high - low) for open_, high, low, close in zip(
            history["open"], history["high"], history["low"], history["close"]
        )]
        decile_threshold = len(high_low_diffs) // 10
        top_decile_high_low_diffs = sorted(high_low_diffs)[-decile_threshold:]
        bottom_decile_high_low_diffs = sorted(high_low_diffs)[:decile_threshold]

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            high_low_diffs_symbol = [
                hld for idx, hld in enumerate(high_low_diffs) if history["symbol"][idx] == symbol
            ]
            if len(high_low_diffs_symbol) < self._window:
                continue
            if max(high_low_diffs_symbol) >= top_decile_high_low_diffs[-1]:
                picks.append(symbol)
            elif min(high_low_diffs_symbol) <= bottom_decile_high_low_diffs[0]:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 0.1
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