from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 20-day simple moving "
        "average of daily returns and the recent price action compared to its 50-day high. By "
        "combining these signals, we aim to identify stocks with both a positive trend and "
        "relatively strong current performance."
    )

    def __init__(self, window_20: int = 20, window_50: int = 50) -> None:
        self._window_20 = window_20
        self._window_50 = window_50

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_20, self._window_50))
        if history.height < max(self._window_20, self._window_50):
            return Signal(information_available_at=stamp, weights={})

        signals: list[str] = []

        for symbol in view.symbols:
            hist = history.filter(pl.col("symbol") == symbol)
            closes = hist.select(
                pl.col("adj_close").tail(self._window_20).alias("closes"),
                pl.col("adj_close").tail(self._window_50).alias("high_50")
            )

            if closes.is_empty():
                continue

            returns_20 = (closes["closes"].to_list()[1:] / closes["closes"].shift(1).to_list() - 1.0)
            sma_20 = sum(returns_20) / self._window_20
            high_50 = max(closes["high_50"].to_list())

            if closes["adj_close"][-1] > high_50 * 0.98 and sma_20 >= 0:
                signals.append(symbol)

        weights = {s: 1.0 / len(signals) for s in signals} if signals else {}
        return Signal(
            information_available_at=stamp, weights={**weights}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max())["session_date"].to_list()[0]
    assert isinstance(newest, date)
    return newest