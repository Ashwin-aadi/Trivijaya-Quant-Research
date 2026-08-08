from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression occurs when a stock's price moves within a smaller range than its "
        "historical volatility. This suggests that the market is consolidating and may soon "
        "break out of this range, providing an opportunity for traders to profit from the "
        "expected increase in volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        compressed: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            df = history.filter(pl.col("symbol") == symbol)
            opens = [float(v) for v in df.select("open").to_series().to_list()]
            closes = [float(v) for v in df.select("close").to_series().to_list()]

            # Calculate the range of each day's price action
            ranges = [(high - low) for high, low in zip(closes, opens)]
            mean_range = sum(ranges) / len(ranges)

            if mean_range < 1.0:  # Assuming a typical daily range for NIFTY constituents
                compressed[symbol] = mean_range

        if not compressed:
            return Signal(information_available_at=stamp, weights={})

        symbol = max(compressed, key=lambda s: compressed[s])
        weight = 1.0 / len(compressed)
        return Signal(
            information_available_at=stamp, weights={symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest