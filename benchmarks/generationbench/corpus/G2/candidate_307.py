from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a security's price fluctuates more widely within its "
        "trading range. This can indicate increased volatility and may be associated with upcoming"
        "price movements, potentially offering trading opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        symbols_with_data = set(latest_closes.keys()).intersection(set(history["symbol"].to_list()))

        signals: dict[str, float] = {}
        for symbol in symbols_with_data:
            df = history.filter(pl.col("symbol") == symbol)
            high_low_ratio = (df.select(pl.col("high").max() / pl.col("low").min()) - 1.0).item()
            if high_low_ratio > 2:  # Assuming a threshold for range compression
                signals[symbol] = 1.0

        weights = {s: weight for s, weight in signals.items()}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest