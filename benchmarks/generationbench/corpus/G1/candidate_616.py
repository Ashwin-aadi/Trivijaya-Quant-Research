from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class TrailingReversion(Strategy):
    rationale = (
        "Trailing reversion strategies identify stocks that have deviated significantly from "
        "their recent price levels and are likely to revert. This strategy aims to capture "
        "such deviations by comparing current prices against a trailing average."
    )

    def __init__(self, window: int = 50, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 2)

        if history.height < self._window + 2:
            return Signal(information_available_at=stamp, weights={})

        means: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            if symbol_history.is_empty():
                continue
            mean_price = symbol_history.select(
                (pl.col("adj_close").mean()).alias("mean_price")
            ).item()
            means[symbol] = mean_price

        scores: dict[str, float] = {}
        for symbol in view.symbols:
            symbol_closes = history.filter(pl.col("symbol") == symbol)
            if symbol_closes.is_empty():
                continue
            latest_close = symbol_closes.select(pl.col("adj_close").last()).item()
            mean_price = means[symbol]
            score = abs(latest_close - mean_price) / mean_price
            scores[symbol] = score

        sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        top_symbols = list(sorted_scores.keys())[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest