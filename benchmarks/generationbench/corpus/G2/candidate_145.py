from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversionShortHorizon(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of asset prices to revert to their "
        "historical means after deviating significantly. In periods of high volatility or extreme "
        "price movement, this strategy aims to identify and capitalize on such reversions."
    )

    def __init__(self, window: int = 5, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            z_scores = _calculate_z_scores(closes[symbol].to_list(), window=self._window)
            if len(z_scores) < self._window:
                continue

            latest_z_score = z_scores[-1]
            if abs(latest_z_score) > self._z_score_threshold:
                signals[symbol] = 0.0

        return Signal(
            information_available_at=stamp,
            weights={s: 1.0 for s in signals.keys()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_z_scores(prices: list[float], window: int) -> list[float]:
    mean_prices = [sum(prices[i : i + window]) / window for i in range(len(prices) - window + 1)]
    mean_price = sum(mean_prices) / len(mean_prices)
    std_dev_prices = (sum([(p - mean_price) ** 2 for p in prices]) / len(prices)) ** 0.5
    return [(p - mean_price) / std_dev_prices for p in prices[-window:]]