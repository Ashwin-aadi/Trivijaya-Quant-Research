from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "To exploit breakout continuation in the Indian market, we identify stocks that have "
        "recently broken out of consolidation patterns and confirm with strong volume. These "
        "stocks are likely to continue their upward trend due to psychological and technical "
        "factors."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_dict()["adj_close"]]
            close_price = float(view.latest_close()[symbol])
            bollinger_high = history.with_columns(
                (pl.col("high") + 2 * pl.col("std").shift(-1)).alias("bollinger_high")
            ).select("bollinger_high").with_column(pl.lit(close_price).alias("close")).to_dict()["bollinger_high"]
            if close_price > bollinger_high[-1] and values[-1] >= max(values):
                volume_ratio = close_price / pl.col("volume").mean().over(pl.date_range(history["session_date"].min(), history["session_date"].max(), interval="20d")).shift(-1)
                if volume_ratio >= 1.5:
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