from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Relative Strength (RS) strategy selects the top N performing stocks based on their "
        "strength relative to the broader market. This approach assumes that strong performers "
        "in the market will continue to outperform over the medium term."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        strength_scores = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_prices = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_prices) < self._window:
                continue
            avg_close_price = sum(close_prices[-self._window:]) / self._window
            strength_score = max((close / avg_close_price - 1.0 for close in close_prices), default=0)
            strength_scores[symbol] = strength_score

        top_symbols = sorted(strength_scores, key=lambda k: strength_scores[k], reverse=True)[:self._top_n]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest