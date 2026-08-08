from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a significant breakout in one direction, prices often reverse and continue "
        "towards the opposite extreme. This strategy looks for symbols that have just "
        "broken out to the upside and are now close to their 20-day low, suggesting they may "
        "reverse and move higher."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1).sort("session_date")
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        for symbol in view.symbols:
            hist = history.select(
                pl.col("symbol") == symbol
            ).sort("session_date").to_pandas()

            # Check if the latest close is a breakout (upward)
            if (
                len(hist) > 1
                and float(hist.iloc[-1]["close"]) >= float(hist.iloc[-2]["adj_close"])
            ):
                # Find the lowest price in the last window days excluding today's low
                recent_low = min(float(v) for v in hist["low"].to_list()[:-1])
                latest_close = float(hist.iloc[-1]["close"])

                # Check if it is close to its 20-day low
                if (
                    1.05 * recent_low <= latest_close < 1.3 * recent_low
                    and (latest_close - recent_low) / recent_low < 0.25
                ):
                    return Signal(
                        information_available_at=stamp,
                        weights={symbol: 1.0},
                    )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest