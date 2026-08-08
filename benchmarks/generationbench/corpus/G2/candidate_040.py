from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price reversion occurs when a stock's price reverts to its historical mean level "
        "after deviating from it. This strategy aims to identify stocks that have "
        "deviated significantly from their mean and are likely to revert back."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 1.5) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history.columns]
        means: dict[str, float] = {}
        stds: dict[str, float] = {}

        # Calculate mean and standard deviation
        for symbol in symbols:
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            mean = sum(prices) / len(prices)
            means[symbol] = mean

            std_dev = (sum((p - mean) ** 2 for p in prices) / len(prices)) ** 0.5
            stds[symbol] = std_dev

        # Calculate z-scores and select symbols with high absolute z-scores
        picks: list[str] = []
        for symbol in symbols:
            latest_close = float(view.latest_close()[symbol])
            mean = means[symbol]
            std_dev = stds[symbol]

            if std_dev == 0:
                continue

            z_score = (latest_close - mean) / std_dev
            if abs(z_score) > self._z_score_threshold:
                picks.append(symbol)

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