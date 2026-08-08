from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and may be setting up "
        "for a breakout. This strategy identifies symbols with reduced volatility over a period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            opens = [float(o) for o in history[symbol + "_open"].to_list()]
            closes = [float(c) for c in history[symbol + "_close"].to_list()]

            # Calculate range (high - low) and price change (close - open)
            ranges = [(h - l, c - o) for h, l, c, o in zip(
                history[symbol + "_high"].to_list(), 
                history[symbol + "_low"].to_list(), 
                closes, opens
            )]

            # Filter out null values and calculate range compression ratio
            ranges = [(r, p_change) for r, p_change in ranges if not (pl.col(symbol + "_high").is_null() | pl.col(symbol + "_low").is_null())]
            
            if len(ranges) < self._window:
                continue

            # Calculate mean and standard deviation of range
            avg_range = sum([r[0] for r in ranges]) / len(ranges)
            std_dev_range = (sum([(r[0] - avg_range)**2 for r in ranges])/len(ranges))**0.5
            
            if avg_range > 0:
                compression_ratio = std_dev_range / avg_range
                if compression_ratio < 0.3:  # Threshold can be adjusted
                    picks.append(symbol)

        picks = picks[:5]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest