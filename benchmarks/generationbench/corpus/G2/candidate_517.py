from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that security prices and investment returns will eventually "
        "move back towards the long-term mean. In a short-horizon context, recent extreme "
        "deviations from the mean are likely to revert, creating an opportunity for profit."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        closes = pl.DataFrame({s: history[s]["adj_close"].to_list() for s in symbols})
        mean_close = closes.mean(axis=1).alias("mean")
        std_dev = (closes.std(axis=1) / self._threshold).alias("std")

        # Identify symbols that have deviated from the mean by more than the threshold
        conditions = [
            (closes[symbol] - mean_close < -std)
            | (closes[symbol] - mean_close > std)
            for symbol, std in zip(symbols, std_dev.to_list())
        ]
        deviations = pl.concat(conditions).alias("deviation")

        # Filter symbols with significant deviation
        filtered_symbols = closes.select(
            [pl.all(), deviations]
        ).filter(pl.col("deviation")).columns[0][1:-2]

        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp, weights=dict(zip(filtered_symbols, [weight] * len(filtered_symbols)))
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest