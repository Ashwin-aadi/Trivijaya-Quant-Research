from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts out of recent trading ranges are identified using daily high and low prices. "
        "These breakouts are confirmed by increased volume relative to a 20-day moving average or "
        "crossing a 20-day moving average, with positions entered at the next open after the breakout. "
        "Positions are exited if the price reverts within ±3% of the breakout level, forms a new consolidation pattern, or if volume decreases significantly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 30)
        if history.is_empty() or history.height < self._window + 30:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            hist = history.select(
                pl.col("symbol").eq(symbol).alias("match"),
                pl.col("high"),
                pl.col("low"),
                (pl.col("volume") / pl.col("volume").rolling_mean(window_size=self._window)).alias("vol_ratio"),
                (pl.col("close").rolling_mean(window_size=self._window) - 0.03 * pl.col("close")).alias("mean_minus_3pct")
            ).filter(pl.col("match"))

            if hist.height < self._window:
                continue

            high_breakout = hist.filter(
                (pl.col("high") > pl.col("close").shift(1)) & 
                (pl.col("volume") >= 2 * pl.col("vol_ratio"))
            )
            low_breakout = hist.filter(
                (pl.col("low") < pl.col("close").shift(1)) & 
                (pl.col("volume") >= 2 * pl.col("vol_ratio"))
            )

            mean_high = high_breakout.select(pl.col("high").mean()).row(0)[0]
            mean_low = low_breakout.select(pl.col("low").mean()).row(0)[0]

            if not mean_high.is_nan() and history.filter(
                (pl.col("symbol") == symbol) & 
                ((pl.col("close") < mean_minus_3pct))
            ).height > 0:
                continue

            if high_breakout.height > 0 or low_breakout.height > 0:
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest