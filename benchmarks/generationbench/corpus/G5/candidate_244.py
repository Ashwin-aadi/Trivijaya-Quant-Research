from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion is a principle that after a significant move in price, the "
        "price will tend to revert back towards its mean. This strategy identifies stocks"
        "that have moved significantly and bets on their reversion."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not all(symbol in closes.columns for symbol in view.symbols):
            return Signal(information_available_at=stamp, weights={})

        reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            mean_price = sum(values) / len(values)
            deviation_from_mean = abs(values[-1] - mean_price)
            reversion_scores[symbol] = deviation_from_mean

        sorted_scores = sorted(reversion_scores.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, score in sorted_scores if score > 0.5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest