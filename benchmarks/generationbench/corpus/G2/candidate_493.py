from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Mean reversion strategies exploit the tendency of stock prices to return to their mean "
        "over time. In a short horizon, stocks that have deviated significantly from their recent "
        "mean can be expected to revert towards it."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        means = (
            history.group_by("symbol")
            .agg(pl.col("adj_close").mean().alias("mean"))
            .select(["symbol", "mean"])
        )
        closes = view.closes(lookback=self._window)
        symbol_means = means.select("symbol").to_list()[0]

        def within_mean_reversion_threshold(symbol: str) -> bool:
            mean = means.filter(pl.col("symbol") == symbol)["mean"].item()
            latest_close = float(closes[symbol].drop_nulls().last())
            return abs(latest_close - mean) / mean > self._threshold

        symbols_within_threshold = [
            s for s in view.symbols if within_mean_reversion_threshold(s)
        ]

        weights: dict[str, float] = {}
        if symbols_within_threshold:
            weight_per_symbol = 1.0 / len(symbols_within_threshold)
            for symbol in symbols_within_threshold:
                weights[symbol] = weight_per_symbol
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest