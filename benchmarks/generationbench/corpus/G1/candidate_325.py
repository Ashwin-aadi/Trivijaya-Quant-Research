from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that if a stock has moved significantly away from its mean "
        "price over the last 10 days, it is likely to move back towards that mean. This can "
        "be used as a basis for short-term trading strategies."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        means = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_price = sum(values) / len(values)
            latest_close = float(view.latest_close()[symbol])
            deviation = (latest_close - mean_price) / mean_price
            if deviation < -0.1:
                means.append(symbol)

        weights = {s: 1.0 / len(means) for s in means}
        return Signal(
            information_available_at=stamp, weights={**weights, "NIFTY 100": 1.0 - sum(weights.values())}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest