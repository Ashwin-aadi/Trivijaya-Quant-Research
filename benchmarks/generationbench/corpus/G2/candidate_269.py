from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation of a breakout is often observed when a stock or index moves past its "
        "previous high or low, but does not immediately reverse. This suggests that the breakout "
        "was genuine and could continue to trend in the direction of the breakout."
    )

    def __init__(self, window: int = 20, lookback: int = 40) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback + 1:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            df = history.select(
                pl.col("symbol"),
                pl.col("session_date"),
                (pl.col("high") - pl.col("low")).alias("range_width"),
                pl.last().over("symbol").alias("last_close"),
            )
            breakout_highs = df.filter(pl.col("session_date") >= date(2020, 1, 1))\
                               .sort("session_date", descending=False).group_by("symbol")\
                               .agg((pl.max("high")).alias("breakout_high"))
            if breakout_highs.height == 0:
                continue
            range_width = (breakout_highs["breakout_high"] - df.select(
                pl.col("last_close").filter(pl.col("session_date") < date(2020, 1, 1))
            ).item())
            if range_width > self._window * 0.05:  # Allow a small margin of error
                breakout_symbols.append(symbol)

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest