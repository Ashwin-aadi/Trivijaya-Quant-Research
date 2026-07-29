from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion assumes that prices which deviate significantly from "
        "their historical average will revert to it. This strategy aims to identify stocks "
        "that have moved too far from their 10-day moving average and bet on a return towards "
        "the mean."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        symbols = history["symbol"].to_list()

        mean = sum(closes) / len(closes)
        deviations = [(sym, abs(close - mean)) for sym, close in zip(symbols, closes)]
        sorted_deviations = sorted(deviations, key=lambda x: x[1], reverse=True)

        picks = [symb for symb, _ in sorted_deviations[:5]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest