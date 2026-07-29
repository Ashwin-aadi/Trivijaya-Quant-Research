from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion2d(Strategy):
    rationale = (
        "Mean reversion is a strategy that seeks to exploit the tendency of financial "
        "assets to return to their mean price level over time. In this implementation, we "
        "identify symbols that have deviated significantly from their recent average and "
        "expect them to revert to the mean."
    )

    def __init__(self, window: int = 2, threshold: float = 0.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_reversion_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            window_values = adj_close_series[-self._window:]
            mean_value = sum(window_values) / self._window
            latest_price = adj_close_series[-1]
            if abs(latest_price - mean_value) > self._threshold * mean_value:
                mean_reversion_scores[symbol] = (latest_price - mean_value) / mean_value

        symbols_to_trade = sorted(mean_reversion_scores, key=mean_reversion_scores.get, reverse=True)
        weight = 1.0 / len(symbols_to_trade) if symbols_to_trade else 0
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_to_trade}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest