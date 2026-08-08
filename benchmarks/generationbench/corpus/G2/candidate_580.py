from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price levels tend to revert to their mean over time. By identifying symbols that have "
        "recently deviated from their historical price ranges and are now trading close to or at "
        "their 20-day moving average, we can generate long positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_history = {
            sym: history.select(
                pl.col("session_date"), pl.col("adj_close").alias(f"{sym}_close")
            )
            .with_columns((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
            .sort("session_date")
            for sym in view.symbols
        }

        mean_reversion = {
            symbol: float(values[-1]) < (float(values[:-2].mean()) * 0.95)
            for symbol, values in symbol_history.items()
            if len(values) > self._window
        }

        signals = {symbol: weight for symbol, weight in mean_reversion.items() if weight}

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        total_weight = sum(signals.values())
        adjusted_weights = {k: v / total_weight for k, v in signals.items()}
        return Signal(
            information_available_at=stamp, weights=adjusted_weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest