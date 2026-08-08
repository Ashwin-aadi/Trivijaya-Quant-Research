from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices and financial returns will eventually "
        "reverse from a high or low. In the short term, this can be exploited by identifying "
        "stocks that have deviated significantly from their mean price over recent history."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_prices = [float(v) for v in history[symbol].to_list()]
            mean_price = sum(close_prices[-self._window:]) / self._window
            current_close = float(history.select(pl.col("adj_close").last()).to_series().item())
            deviation = abs(current_close - mean_price)
            if deviation > self._threshold * (mean_price / self._window):
                symbols.append(symbol)

        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(
            information_available_at=stamp, weights={k: v for k, v in weights.items() if v != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().item()
    assert isinstance(newest, date)
    return newest