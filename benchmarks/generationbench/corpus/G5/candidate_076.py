from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceReversion(Strategy):
    rationale = (
        "Price reversion strategies look for securities that have moved away from their "
        "historical price levels and predict a return to those levels. This strategy uses a "
        "trailing reference point to identify such opportunities."
    )

    def __init__(self, window: int = 60, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        if closes.height == 0:
            return Signal(information_available_at=stamp, weights={})

        symbol = list(closes.columns)[-1]
        if symbol not in history.symbol.to_list():
            return Signal(information_available_at=stamp, weights={})

        recent_close = float(closes[symbol].to_list()[-1])
        historical_prices = (
            history.filter(pl.col("symbol") == symbol)
                   .select(pl.col("adj_close"))
                   .to_pandas()
                   .iloc[:, 0]
        )

        if len(historical_prices) < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price = historical_prices.mean()
        std_dev = historical_prices.std()

        z_score = (recent_close - mean_price) / std_dev
        if abs(z_score) > self._threshold:
            return Signal(
                information_available_at=stamp,
                weights={symbol: 1.0},
            )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest