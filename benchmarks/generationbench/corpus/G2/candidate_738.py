from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength tend to outperform the broader market over time. "
        "This is based on the assumption that stocks in strong sectors or with a history of "
        "outperformance will continue to do well."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_vertical()
        symbols_with_data = set(mean_close.columns) & set(view.symbols)

        strength_scores: dict[str, float] = {}
        for symbol in symbols_with_data:
            if view.closes(symbol)[-1] > mean_close[symbol][-1]:
                strength_scores[symbol] = 1.0
            else:
                strength_scores[symbol] = 0.0

        sorted_strengths = sorted(strength_scores.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, score in sorted_strengths if score == 1.0]

        weight_per_symbol = 1.0 / len(top_symbols) if top_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest