from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price levels that revert to their trailing mean suggest that the current price "
        "deviation from its historical average is likely to correct. This behavior can be "
        "exploited for profit if there are enough periods in history to establish a reliable "
        "mean."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trailing_means: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            mean_value = sum(values) / self._window
            trailing_means[symbol] = mean_value

        weights: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns or symbol not in trailing_means:
                continue
            close_price = float(closes[symbol].last())
            if close_price < 0.95 * trailing_means[symbol]:
                weights[symbol] = -1.0
            elif close_price > 1.05 * trailing_means[symbol]:
                weights[symbol] = 1.0

        # Ensure the sum of non-zero weights is 1, otherwise weight remaining assets as cash.
        total_weight = sum(weights.values())
        if total_weight != 0:
            for symbol in weights:
                weights[symbol] /= total_weight
        else:
            weights.clear()

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest