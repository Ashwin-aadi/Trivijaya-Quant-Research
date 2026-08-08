from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two simple characteristics: recent price momentum and "
        "volume anomalies. Price momentum indicates strong buying pressure, while volume "
        "anomalies suggest increased interest in a stock."
    )

    def __init__(self, window: int = 10, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["close"].to_list()
        momentum = (closes[-1] - closes[0]) / closes[0]
        volume_history = view.history(lookback=self._window).select(
            pl.col("symbol").alias("symbol"), pl.col("volume")
        )
        volume_ratio = (
            volume_history.select(pl.col("volume").mean().alias("avg_volume"))
            .with_columns((pl.col("volume") / pl.col("avg_volume")).alias("ratio"))
            .filter(pl.col("ratio") > self._threshold)
            .select(pl.col("symbol"))
        )
        selected_symbols = volume_ratio["symbol"].to_list()

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}
        if momentum > 0.1:
            weights[selected_symbols[0]] += 0.5

        return Signal(
            information_available_at=stamp, weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest