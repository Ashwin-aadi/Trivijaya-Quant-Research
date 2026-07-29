from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy selects stocks that have shown the strongest price momentum over a "
        "recent period. The idea is to capitalize on the tendency of strong performers to "
        "continue performing well."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["close"].to_list()]
        momentum_scores = {symbol: closes[i] / closes[i - self._window] for i, symbol in enumerate(view.symbols)}
        sorted_symbols = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        
        picks: list[str] = [symbol for symbol, _ in sorted_symbols[:self._top_n]]
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