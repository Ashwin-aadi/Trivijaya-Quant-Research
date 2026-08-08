from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks with the highest relative strength against the NIFTY 100 index "
        "can help identify outperformers. The idea is that stocks with higher returns "
        "relative to the market may offer better risk-adjusted returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        nifty_returns = (closes["^NSEI"] / closes["^NSEI"].shift(1) - 1.0).to_list()[self._window:]
        symbol_returns: dict[str, float] = {}
        
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window + 1:
                continue
            last_close = float(values[-1])
            mean_return = sum((close / last_close - 1.0 for close in values[:self._window]), start=0.0)
            symbol_returns[symbol] = mean_return

        sorted_returns = sorted(symbol_returns.items(), key=lambda x: x[1], reverse=True)[:5]
        top_symbols = [symb for symb, _ in sorted_returns]

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