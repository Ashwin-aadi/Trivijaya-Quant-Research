from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long run. "
        "This is due to a combination of lower risk and potentially higher risk-adjusted returns. "
        "By tilting the portfolio towards low-volatility stocks, one can capture this performance difference."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        volatilities = {
            symbol: float(
                (history[symbol].to_list()[1:] / history[symbol].shift(1).to_list() - 1.0)
                .abs()
                .mean()
            )
            for symbol in symbols
        }
        sorted_symbols = [symbol for _, symbol in sorted(volatilities.items(), key=lambda item: item[1])]
        
        weights = {sorted_symbols[i]: (2 - i) / len(sorted_symbols) for i in range(len(sorted_symbols))}
        return Signal(
            information_available_at=stamp, 
            weights={s: weights.get(s, 0.0) for s in view.symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest