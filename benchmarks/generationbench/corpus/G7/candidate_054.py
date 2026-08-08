from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RSI50d(Strategy):
    rationale = (
        "The Relative Strength Index (RSI) of 50 days helps in identifying overbought or oversold "
        "conditions relative to the broad market. This can signal potential entry and exit points "
        "for profitable trades."
    )

    def __init__(self, window: int = 50, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        rsi_values = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            avg_gain = sum(max(v - values[i-1], 0.0) for i, v in enumerate(values)) / self._window
            avg_loss = abs(sum(min(v - values[i-1], 0.0) for i, v in enumerate(values))) / self._window
            if avg_loss == 0:
                rsi_values[symbol] = 100.0
            else:
                rsi_values[symbol] = 100.0 - (100.0 / (1 + avg_gain / avg_loss))

        ranked_symbols = sorted(rsi_values, key=lambda s: rsi_values[s], reverse=True)
        top_n_symbols = ranked_symbols[:self._top_n]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest