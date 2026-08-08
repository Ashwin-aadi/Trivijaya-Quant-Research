from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Reversion to the mean is a common financial principle suggesting that prices and returns "
        "tend to move towards an average over time. By identifying stocks whose recent performance "
        "is far from their historical average, we can predict a reversion back toward this mean."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().to_list()[0]
        std_dev_close = closes.std().to_list()[0]

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            recent_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()[-self._window :]]
            z_score = (recent_closes[-1] - mean_close) / std_dev_close
            if abs(z_score) > 2.0:  # Threshold for reversion
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        total_weight = sum(signals.values())
        weight_per_symbol = {s: w / total_weight for s, w in signals.items()}
        return Signal(
            information_available_at=stamp, weights=weight_per_symbol
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest