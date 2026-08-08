from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion suggests that security prices and financial returns eventually "
        "return to the long-term mean. In a short horizon, extreme deviations from this mean are "
        "likely to be followed by a return toward it."
    )

    def __init__(self, window: int = 10, threshold_factor: float = 2.0) -> None:
        self._window = window
        self._threshold_factor = threshold_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = float(closes.mean().select(pl.col("adj_close").mean()).to_series()[0])
        std_close = float(closes.std().select(pl.col("adj_close").std()).to_series()[0])

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if abs(values[-1] - mean_close) >= self._threshold_factor * std_close:
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
    newest = visible.select(pl.col("session_date").max()).to_series()[0]
    assert isinstance(newest, date)
    return newest