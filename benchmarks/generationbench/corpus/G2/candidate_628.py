from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which deviate significantly from their "
        "historic average will revert back. We can use a trailing reference price level, such "
        "as a 50-day simple moving average (SMA), to identify stocks that are overbought or oversold."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        sma_values = (
            history.group_by("symbol")
            .agg((pl.col("adj_close").mean().alias("sma")))
            .with_columns(
                (pl.col("adj_close") / pl.col("sma") - 1.0).alias("reversion_signal")
            )
        )

        if sma_values.is_empty():
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []
        for symbol in view.symbols:
            if symbol not in sma_values.columns:
                continue
            reversion_signal = float(sma_values[symbol]["reversion_signal"])
            if abs(reversion_signal) > 0.2:  # Threshold for overbought/oversold condition
                signals.append(symbol)

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest