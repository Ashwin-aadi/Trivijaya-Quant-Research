from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates reduced volatility and increased consolidation. "
        "Stocks that have been consolidating might be due for a breakout or correction. "
        "By identifying such stocks, we can potentially profit from the upcoming price movement."
    )

    def __init__(self, window: int = 30, threshold: float = 0.15) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.select("symbol").unique().to_list():
                continue
            high_low_diff = (
                history.filter(pl.col("symbol") == symbol)
                .select((pl.col("high") - pl.col("low")).alias("range"))
                .item()
            )
            recent_high_low_diff = (
                history.filter(pl.col("symbol") == symbol)
                .sort("session_date", descending=True)
                .head(10)
                .select(
                    (pl.col("high").max() - pl.col("low").min()).alias("recent_range")
                )
                .item()
            )
            if recent_high_low_diff / high_low_diff <= self._threshold:
                picks.append(symbol)

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        # Adjusting the number of picked symbols dynamically
        top_n = min(len(picks), 10)
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks[:top_n]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest