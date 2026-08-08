from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionStrategy(Strategy):
    rationale = (
        "This strategy aims to capture dispersion or range compression in the Indian equity market "
        "through mean reversion. By entering positions when the daily High-Low spread drops below its "
        "20-day moving average and exiting after 5 trading days or when the spread exceeds a threshold, "
        "the strategy seeks to profit from price reversions while managing risk."
    )

    def __init__(self, window: int = 20, high_low_threshold: float = 1.1) -> None:
        self._window = window
        self._high_low_threshold = high_low_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 5)

        if history.height < self._window + 5:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        high_low_spread = [high - low for high, low in zip(closes[1:], closes[:-1])]
        mean_spread = sum(high_low_spread) / self._window

        recent_high_low_spread = history.sort("session_date").tail(self._window)["adj_close"].to_list()
        recent_high_low_spread = [recent_high_low_spread[i] - recent_high_low_spread[i + 1] for i in range(len(recent_high_low_spread) - 1)]

        if all(spread < mean_spread * self._high_low_threshold for spread in recent_high_low_spread):
            symbols = view.symbols
            weight = 1.0 / len(symbols)
            return Signal(
                information_available_at=stamp,
                weights={s: weight for s in symbols},
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest