from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the fact that securities with positive recent "
        "returns tend to continue outperforming. This strategy buys those top performers and "
        "sells bottom performers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (closes[closes.columns[:-1]] / closes[closes.columns[1:]] - 1.0).fill_null(0)

        # Filter out symbols with insufficient data
        filtered_returns = {symbol: values.to_list() for symbol, values in returns.items()}
        symbols_with_data = [s for s, v in filtered_returns.items() if len(v) == self._window]
        
        if not symbols_with_data:
            return Signal(information_available_at=stamp, weights={})

        # Compute mean returns
        mean_returns = {symbol: float(sum(values)) / self._window for symbol, values in filtered_returns.items()}
        
        top_symbols = sorted(mean_returns.keys(), key=lambda s: -mean_returns[s])[:self._top_n]
        bottom_symbols = sorted(mean_returns.keys(), key=lambda s: mean_returns[s])[:self._top_n]

        # Create weights
        weight_top = 1.0 / len(top_symbols)
        weight_bottom = -1.0 / len(bottom_symbols)

        weights = {s: weight_top for s in top_symbols}
        for s in bottom_symbols:
            if s not in weights.keys():
                weights[s] = weight_bottom

        return Signal(
            information_available_at=stamp, weights={symbol: weights.get(symbol, 0.0) for symbol in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest