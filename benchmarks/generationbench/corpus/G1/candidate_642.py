from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the NIFTY 100 index "
        "helps identify outperformers. This strategy assumes that such stocks are more likely "
        "to continue their upward trend."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate relative strength
        symbol_closes = {symbol: [float(v) for v in closes[symbol].drop_nulls().to_list()] for symbol in view.symbols}
        nifty_close = symbol_closes["^NIFTY 100"]
        rel_strengths = [(symbol_closes[symbol][-1] / nifty_close[-1]) - 1.0 for symbol in view.symbols]
        
        top_symbols = [view.symbols[i] for i, _ in sorted(zip(range(len(rel_strengths)), rel_strengths), key=lambda x: x[1], reverse=True)[:3]]
        weight = 1.0 / len(top_symbols)
        
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest