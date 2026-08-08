from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Price levels revert to the mean over time. By identifying price movements that "
        "are far from recent means, we can capitalize on potential reversion to these means."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_price: float = closes.select(
            pl.col("adj_close").mean()
        ).to_series().item()
        std_dev: float = closes.select(
            (pl.col("adj_close") - pl.col("adj_close").mean()).stddev()
        ).to_series().item()

        reversion_signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            last_price: float = view.latest_close()[symbol]
            z_score = (last_price - mean_price) / std_dev
            if abs(z_score) > 1.5:
                reversion_signals[symbol] = -0.2 + 0.4 * z_score

        if not reversion_signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(abs(weight) for weight in reversion_signals.values())
        adjusted_weights: dict[str, float] = {
            symbol: weight / total_weight
            for symbol, weight in reversion_signals.items()
        }

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in adjusted_weights.items() if weight != 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest