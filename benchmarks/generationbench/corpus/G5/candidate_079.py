from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when the intraday price action within a bar is reduced. "
        "This can indicate increased indecision and potential for reversal or breakout in either direction."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        high_low_diff = (history["high"] - history["low"]).to_list()
        close_open_ratio = ((history["close"] / history["open"]) - 1.0).to_list()

        compression_scores = {
            symbol: sum(
                (high_low_diff[i] < 0.02 and abs(close_open_ratio[i]) > 0.05)
                for i in range(1, self._window + 1)
            )
            / max(self._window - 1, 1)
            for symbol in symbols
        }

        sorted_symbols = [
            (symbol, score) for symbol, score in compression_scores.items() if score > 0.2
        ]
        sorted_symbols.sort(key=lambda x: x[1], reverse=True)

        top_n_symbols = [symbol for symbol, _ in sorted_symbols[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest