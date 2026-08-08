from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Mean reversion occurs when a stock's price returns to its historical average. "
        "By identifying stocks that have deviated significantly from their mean, we can exploit this tendency."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().select(
            pl.col("adj_close").mean().alias("mean")
        ).get_column("mean")[0]
        deviations = [
            (float(closes[symbol].drop_nulls().to_list()[-1]) - mean_close) / mean_close
            for symbol in view.symbols
            if symbol in closes.columns
        ]

        threshold = 0.05  # Consider as candidate if deviation is more than this percentage
        candidates = [symbol for i, symbol in enumerate(view.symbols) if abs(deviations[i]) > threshold]
        
        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in candidates},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest