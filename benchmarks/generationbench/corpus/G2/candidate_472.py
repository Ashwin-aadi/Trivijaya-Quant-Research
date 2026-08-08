from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionTrailing(Strategy):
    rationale = (
        "Price levels revert to a mean over time. This strategy identifies stocks that have "
        "deviated significantly from their trailing average and bets on reversion."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        means: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            closes = [float(v) for v in history[symbol].to_list()]
            mean_close = sum(closes) / len(closes)
            means[symbol] = mean_close

        filtered_symbols = [
            symbol
            for symbol, close in zip(view.symbols, view.closes(lookback=self._window).columns)
            if (abs(close - means[symbol]) > 0.1 * means[symbol])
        ]

        weight = 1.0 / len(filtered_symbols) if filtered_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest