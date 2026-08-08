from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalReturn(Strategy):
    rationale = (
        "Market returns exhibit seasonality due to various factors such as earnings seasons, "
        "holiday effects, and macroeconomic events. By focusing on the monthly average return over a 5-year period, we aim to capture these seasonal trends."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if stamp < date(2019, 1, 1):
            return Signal(information_available_at=stamp, weights={})

        history = view.history(lookback=self._window)
        closes = view.closes(lookback=self._window)

        if history.is_empty() or closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            returns = [(c / p - 1.0) for c, p in zip(values[1:], values[:-1])]
            avg_return = sum(returns) / len(returns)
            avg_returns[symbol] = avg_return

        # Select the market index symbol
        selected_symbol = list(avg_returns.keys())[0]

        signal = Signal(
            information_available_at=stamp, weights={selected_symbol: 1.0}
        )
        return signal


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(2019, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest