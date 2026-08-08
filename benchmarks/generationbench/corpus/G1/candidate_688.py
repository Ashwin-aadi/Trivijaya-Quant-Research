from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of prices to return to their mean "
        "over time. By identifying symbols that have moved significantly from their trailing "
        "average, we can take positions in those that are expected to revert."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        mean_prices = (
            history.group_by("symbol")
                   .agg((pl.col("adj_close").mean()).alias("trailing_mean"))
        )
        latest_closes = view.closes(lookback=self._window)

        symbols_with_price = set(mean_prices.select(pl.col("symbol")).to_list()[0])
        valid_symbols = [s for s in symbols_with_price if s in latest_closes.columns]

        reversion_scores: list[float] = []
        selected_symbols: list[str] = []

        for symbol in valid_symbols:
            mean_price = float(mean_prices.filter(pl.col("symbol") == symbol)[
                "trailing_mean"
            ])
            latest_close = float(latest_closes[symbol])
            score = (latest_close - mean_price) / mean_price

            if abs(score) > 0.1:  # Threshold for reversion signal
                selected_symbols.append(symbol)
                reversion_scores.append(score)

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        total_score = sum(reversion_scores)
        weight_per_symbol = {s: score / total_score for s, score in zip(selected_symbols, reversion_scores)}
        return Signal(
            information_available_at=stamp, weights=weight_per_symbol
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest