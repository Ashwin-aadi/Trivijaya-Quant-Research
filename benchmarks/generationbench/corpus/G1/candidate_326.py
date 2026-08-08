from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySFTrendFollowing(Strategy):
    rationale = (
        "Trend following based on volatility scaling. Identifies symbols with significant"
        "price movements and allocates capital accordingly to capitalize on the momentum."
    )

    def __init__(self, window: int = 20, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_series = [float(v) for v in history[symbol].to_list()]
            change = (
                pl.Series(close_series).rolling_std(window=self._window)
                / (pl.Series(close_series).shift(1) - pl.Series(close_series)).abs()
            ).mean().item()
            if change > self._threshold:
                trends[symbol] = 1.0

        weights = {s: w for s, w in trends.items() if w > 0}
        if not weights:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp,
            weights={
                symbol: value
                for symbol, value in weights.items()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().to_date()
    assert isinstance(newest, date)
    return newest