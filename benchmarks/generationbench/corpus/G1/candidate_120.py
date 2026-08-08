from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum invests in stocks that have performed well relative to the market "
        "over a recent period. This strategy captures the idea that past winners are likely to outperform "
        "in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = [float(v) for v in closes[view.symbols[-1]].drop_nulls().to_list()]
        recent_changes = [
            (latest_closes[i] - float(closes[symbol][0])) / float(closes[symbol][0])
            for i, symbol in enumerate(view.symbols)
        ]
        
        top_symbols = sorted(zip(view.symbols, recent_changes), key=lambda x: x[1], reverse=True)[:5]
        top_symbols = [symbol for symbol, _ in top_symbols]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest