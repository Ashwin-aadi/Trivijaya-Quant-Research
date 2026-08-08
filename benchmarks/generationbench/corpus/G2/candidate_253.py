from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of asset prices to revert to their mean "
        "values over time. In a short horizon, extreme movements in price may indicate a high "
        "probability that the price will move back towards its historical average."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_returns = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            mean_return = sum(values[-self._window :]) / self._window - 1.0
            latest_return = (values[-1] / values[-2] - 1.0) * 100.0
            reversion_signal = abs(latest_return - mean_return)
            if reversion_signal > self._threshold:
                symbol_returns[symbol] = reversion_signal

        symbols_to_trade = [k for k, v in symbol_returns.items() if v >= self._threshold]
        if not symbols_to_trade:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_to_trade)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_trade},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest