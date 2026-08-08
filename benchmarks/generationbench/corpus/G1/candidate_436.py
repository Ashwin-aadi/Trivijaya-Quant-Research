from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Price levels revert to the mean over time. This strategy identifies stocks where "
        "the recent price has deviated significantly from its trailing average and bets on "
        "a reversion."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        prices = {symbol: float(v) for symbol, v in zip(symbols, history["adj_close"].to_list())}
        means = {symbol: price.mean() for symbol, price in prices.items()}
        std_devs = {symbol: price.std() for symbol, price in prices.items()}

        def z_score(price: float, mean: float, std_dev: float) -> float:
            return (price - mean) / std_dev if std_dev > 0 else 0.0

        z_scores = {symbol: z_score(prices[symbol], means[symbol], std_devs[symbol]) for symbol in symbols}
        strong_z_scores = {symbol: value for symbol, value in z_scores.items() if abs(value) >= self._z_score_threshold}

        if not strong_z_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_z_scores)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in strong_z_scores.keys()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest