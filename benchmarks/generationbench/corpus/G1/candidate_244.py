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
        if closes.height < self._window or closes.width == 1:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_dict().popitem()[1]
        std_close = (
            (closes - mean_close).abs()
            .mean()
            .to_dict()
            .popitem()[1]
        )

        symbols_with_high_volatility: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window or abs(values[-1] - mean_close) > 2 * std_close:
                symbols_with_high_volatility.append(symbol)

        weight = 1.0 / len(symbols_with_high_volatility)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols_with_high_volatility}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest