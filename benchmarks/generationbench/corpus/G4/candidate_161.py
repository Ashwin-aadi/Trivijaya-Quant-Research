from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakout continuation strategy targets instances where stock prices break through key support or resistance levels. "
        "Upon a valid breakout, long positions are entered expecting momentum to continue due to herding behavior and profit-taking dynamics."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        high_ma20 = history.group_by("symbol").agg(
            (pl.col("high") / pl.col("high").shift(20).mean()).alias("high_ma20")
        )
        low_ma20 = history.group_by("symbol").agg(
            (pl.col("low") / pl.col("low").shift(20).mean()).alias("low_ma20")
        )

        breakout_highs = (
            high_ma20.join(history, on="symbol", how="inner")
                     .filter((pl.col("high") > 1.01 * pl.col("high_ma20"))
                             & (pl.col("volume") >= view.closes().select(pl.col(view.symbols[0]).mean()).item()))
        )
        breakout_lows = (
            low_ma20.join(history, on="symbol", how="inner")
                     .filter((pl.col("low") < 0.99 * pl.col("low_ma20"))
                             & (pl.col("volume") >= view.closes().select(pl.col(view.symbols[0]).mean()).item()))
        )

        all_breakouts = breakout_highs.vstack(breakout_lows)
        if all_breakouts.is_empty():
            return Signal(information_available_at=stamp, weights={})

        all_breakouts = (
            all_breakouts.sort("high_ma20", descending=True).head(self._top_n)
        )
        weight = 1.0 / len(all_breakouts)
        return Signal(
            information_available_at=stamp, 
            weights={row["symbol"]: weight for row in all_breakouts.iter_rows()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).item()
    assert isinstance(newest, date)
    return newest