from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion is a market phenomenon where prices tend to move back towards the mean. "
        "In short horizons, this can be exploited by going long on underperforming stocks and "
        "shorting outperforming ones."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.groupby("symbol").agg(
            (pl.col("adj_close").mean()).alias("mean")
        )
        latest_closes = view.closes(lookback=self._window)

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in mean_close.height or symbol not in latest_closes.columns:
                continue
            mean = mean_close.get_column("mean").to_list()[0]
            current_close = float(latest_closes[symbol].item())
            z_score = (current_close - mean) / mean_close.get_column("mean").std().item()
            if z_score > 2.0:  # Short
                signals[symbol] = -1.0
            elif z_score < -2.0:  # Long
                signals[symbol] = 1.0

        total_weight = sum(signals.values())
        if total_weight == 0:
            return Signal(information_available_at=stamp, weights={})

        for symbol in signals.keys():
            signals[symbol] /= total_weight
        cash_weight = 1 - sum(signals.values())

        return Signal(
            information_available_at=stamp,
            weights={
                **signals,
                "cash": cash_weight if cash_weight != 0 else 0.0
            }
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest