from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion20d(Strategy):
    rationale = (
        "Short-horizon mean reversion suggests that assets which have deviated significantly "
        "from their historical price range are likely to revert. By identifying such deviations, "
        "we can exploit mean reversion tendencies for profit."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = history.select(pl.col("adj_close").tail(self._window))
        mean_close = recent_closes.mean().item()
        std_close = recent_closes.std().item()

        symbols_above_threshold: list[str] = []
        symbols_below_threshold: list[str] = []

        for symbol in view.symbols:
            if symbol not in history["symbol"]:
                continue
            latest_close = float(history.filter(pl.col("symbol") == symbol).select(pl.last("adj_close")).item())
            z_score = (latest_close - mean_close) / std_close

            if abs(z_score) > self._threshold:
                if z_score > 0:
                    symbols_below_threshold.append(symbol)
                else:
                    symbols_above_threshold.append(symbol)

        weights: dict[str, float] = {}
        for symbol in symbols_above_threshold + symbols_below_threshold:
            weights[symbol] = 1.0 / len(symbols_above_threshold + symbols_below_threshold)

        if not weights:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest