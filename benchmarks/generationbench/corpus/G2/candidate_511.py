from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's high and low prices are tightly packed "
        "compared to its recent average range. This suggests increased volatility and potential"
        "price movement, which can lead to opportunities for traders."
    )

    def __init__(self, window: int = 20, threshold: float = 0.8) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.symbol.unique().to_list():
                continue
            recent_highs = history.filter(pl.col("symbol") == symbol).select(
                pl.col("high").sort(descending=True)
            ).head(self._window)["high"].to_list()
            recent_lows = history.filter(pl.col("symbol") == symbol).select(
                pl.col("low").sort()
            ).head(self._window)["low"].to_list()

            if len(recent_highs) < self._window or len(recent_lows) < self._window:
                continue

            avg_range = sum(high - low for high, low in zip(recent_highs, recent_lows)) / (
                self._window * 2
            )
            recent_range = max(recent_highs) - min(recent_lows)

            if recent_range / avg_range < self._threshold:
                signals[symbol] = 1.0

        if not signals:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(signals)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in signals.keys()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest