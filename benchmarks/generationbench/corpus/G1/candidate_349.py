from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion strategies capitalize on the tendency of financial markets to "
        "revert to historical mean prices. This strategy identifies stocks that have "
        "deviated significantly from their 10-day moving average and bets they will revert."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        means = (history[symbol].mean().item() for symbol in symbols)
        recent_closes = {symbol: float(history[symbol][-1]) for symbol in symbols}
        
        deviations = [
            abs(recent_closes[symbol] - mean) / mean
            for symbol, mean in zip(symbols, means)
        ]
        
        target_symbols = [symbols[i] for i in sorted(range(len(deviations)), key=lambda k: deviations[k])]
        
        if not target_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(target_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in target_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest