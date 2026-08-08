from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion theory suggests that prices and returns eventually move back towards "
        "the mean over a given period. By identifying stocks that have deviated significantly from "
        "their historical price range, we can exploit potential reversions to the mean."
    )

    def __init__(self, window: int = 10, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close: dict[str, float] = {symbol: pl.col(symbol).mean().item() for symbol in view.symbols}
        deviations = [
            (symbol, abs(float(closes[symbol][-1]) - mean_close[symbol]))
            for symbol in view.symbols
        ]
        
        sorted_deviations = sorted(deviations, key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_deviations[:self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest