from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks have historically exhibited better risk-adjusted returns. "
        "By tilting our portfolio towards these stocks, we aim to capture this anomaly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        if len(symbols) < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatility: list[float] = []
        for symbol in symbols:
            close_series = history[symbol].drop_nulls()
            returns = [(close - prev_close) / prev_close
                       for prev_close, close in zip(close_series.shift(1).to_list(), close_series.to_list())]
            if len(returns) < self._window:
                continue
            volatility.append(abs(sum(returns)) / self._window)

        sorted_symbols = [symbol for symbol, _ in sorted(zip(symbols, volatility), key=lambda x: x[1])]
        top_n_symbols = sorted_symbols[:5]

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest