from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "This strategy identifies stocks that have deviated significantly from their "
        "historical price levels and are expected to revert towards those levels over time. "
        "By focusing on mean reversion, the strategy aims to capture undervalued stocks while"
        " managing risk through a stop-loss mechanism."
    )

    def __init__(self, lookback: int = 50, std_dev_threshold: float = -2.0) -> None:
        self._lookback = lookback
        self._std_dev_threshold = std_dev_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback)

        if closes.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].to_list()]
            sma = sum(adj_closes[-self._lookback:]) / self._lookback
            std_dev = (sum((v - sma) ** 2 for v in adj_closes[-self._lookback:]) /
                       self._lookback) ** 0.5

            if std_dev == 0:
                continue

            reversion_score = (adj_closes[-1] - sma) / std_dev
            if reversion_score < self._std_dev_threshold:
                signals[symbol] = 1.0 / len(signals)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp, weights=signals
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest