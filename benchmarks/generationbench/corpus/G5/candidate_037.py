from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion occurs when an asset's price moves towards its historical mean. "
        "Short-horizon mean reversion strategies exploit deviations from this mean by betting "
        "that the price will revert to a long-term average."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        mean_close = (history[symbols] / 100).mean().to_dict(False)
        latest_closes = {symbol: float(close) for symbol, close in view.latest_close().items()}

        deviations = {
            symbol: abs(latest_closes[symbol] - mean_close[symbol])
            for symbol in symbols
        }

        sorted_symbols = [
            s for _, s in sorted(deviations.items(), key=lambda item: item[1], reverse=True)
        ][:5]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest