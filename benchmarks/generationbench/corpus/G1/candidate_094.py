from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "Seasonal effects can significantly impact stock prices. This strategy "
        "exploits historical patterns by identifying stocks that have historically "
        "performed well during certain times of the year."
    )

    def __init__(self, window: int = 120) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        grouped_history = (
            history.groupby("symbol")
                   .agg((pl.col("close") / pl.col("adj_close").shift(self._window) - 1.0).mean().alias("avg_return"))
                   .sort("avg_return", descending=True)
        )

        top_symbols = [row[0] for row in grouped_history.rows()][:5]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={symbol: weight for symbol in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest