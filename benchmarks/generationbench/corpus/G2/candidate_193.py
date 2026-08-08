from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion suggests that prices which have deviated significantly from their "
        "long-term average are likely to return. In a short horizon, if a stock's price has "
        "dropped far below its mean over the past 20 days, it is more likely to increase in "
        "value as it moves back towards its historical norm."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_prices = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            mean_price = float(sum(values) / self._window)
            mean_prices[symbol] = mean_price

        signals: list[str] = []
        for symbol, mean_price in mean_prices.items():
            recent_close = view.latest_close()[symbol]
            if (recent_close - mean_price) / mean_price < -self._threshold:
                signals.append(symbol)

        signals = signals[:5]
        if not signals:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest