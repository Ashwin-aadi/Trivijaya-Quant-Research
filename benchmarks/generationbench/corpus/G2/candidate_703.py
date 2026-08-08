from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression can indicate a stock is in a period of consolidation before a potential "
        " breakout. This strategy aims to identify such stocks by measuring the difference between "
        "the highest and lowest prices over a lookback period relative to their recent price range."
    )

    def __init__(self, window: int = 20, threshold: float = 0.9) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            data = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = [float(v) for v in data["adj_close"].drop_nulls().to_list()]
            high = max(prices)
            low = min(prices)
            recent_high = view.latest_close()[symbol]
            recent_low = view.latest_close()[symbol]

            if (high - low) / (recent_high - recent_low) < self._threshold:
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