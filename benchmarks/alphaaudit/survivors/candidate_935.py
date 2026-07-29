from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "Combining the 20-day closing price momentum with the relative strength "
        "of a stock compared to its peers can provide a more robust signal for entry."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores: dict[str, float] = {}
        relative_strength_scores: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate momentum score as the last price divided by the first price, normalized to 0-1 range.
            start_price = values[0]
            end_price = values[-1]
            momentum_score = (end_price - start_price) / start_price
            if momentum_score > 0:
                momentum_scores[symbol] = momentum_score

        # Calculate relative strength score as the last price divided by the median of all prices.
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            end_price = values[-1]
            median_price = pl.Series(values).median()
            relative_strength_score = end_price / float(median_price)
            if relative_strength_score > 1.05:  # Set a threshold for relative strength.
                relative_strength_scores[symbol] = relative_strength_score

        combined_scores: dict[str, float] = {
            symbol: (momentum_scores.get(symbol, 0) + relative_strength_scores.get(symbol, 0)) / 2
            for symbol in momentum_scores.keys() & relative_strength_scores.keys()
        }

        top_picks: list[str] = sorted(combined_scores, key=combined_scores.get, reverse=True)[: self._top_n]

        if not top_picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest