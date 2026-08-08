from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks "
        "with the highest returns over a lookback period. These stocks are expected "
        "to continue outperforming in the near future."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].to_list()]
            returns = (close_values[-1] - close_values[0]) / close_values[0]
            symbol_returns[symbol] = returns

        top_symbols = sorted(symbol_returns, key=symbol_returns.get, reverse=True)[:5]
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