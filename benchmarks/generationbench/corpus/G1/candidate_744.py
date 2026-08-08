from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well relative to their peers in recent periods to continue outperforming. This "
        "strategy ranks assets based on their returns over a lookback period and allocates "
        "to the top performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        # Calculate relative returns
        relative_returns = {}
        for symbol in view.symbols:
            close_values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_values) < self._window:
                continue
            returns = [(close_values[i] - close_values[i-1]) / close_values[i-1]
                       for i in range(1, len(close_values))]
            relative_returns[symbol] = sum(returns)

        # Rank symbols by their average relative return
        ranked_symbols = sorted(relative_returns.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in ranked_symbols[:self._top_n]]

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