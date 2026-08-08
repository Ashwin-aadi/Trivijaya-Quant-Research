from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionStrategy(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which deviate significantly from their "
        "historical average will eventually return. This strategy seeks to identify such "
        "overbought or oversold conditions and capitalize on them."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        mean_close = closes.mean().item()
        normalized_closes = (closes - mean_close) / mean_close
        z_scores = normalized_closes.zscore().to_list()

        symbols_with_extreme_z_scores: list[str] = []
        for symbol, z_score in zip(view.symbols, z_scores):
            if abs(z_score) > 1.5:
                symbols_with_extreme_z_scores.append(symbol)

        if not symbols_with_extreme_z_scores:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(symbols_with_extreme_z_scores)
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight_per_symbol
                for symbol in symbols_with_extreme_z_scores
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest