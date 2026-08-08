from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Low-volatility stocks are often underpriced and have higher average returns than "
        "high-volatility stocks. By tilting the portfolio towards lower volatility stocks, we "
        "can potentially generate alpha."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            returns = [float(v) for v in (history[symbol]["adj_close"].drop_nulls() / history[symbol]["adj_close"].shift(1).drop_nulls() - 1.0).to_list()]
            if len(returns) < self._window - 1:
                continue
            volatility[symbol] = pl.Series(returns).std()

        sorted_symbols = [s for s, v in sorted(volatility.items(), key=lambda item: item[1])]
        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest