from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies instances where stocks break through previous price levels and continue moving in the direction of the breakout. "
        "By focusing on breakout continuation patterns, we aim to capitalize on the inertia in price movements post-breakout."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        breakout_scores = {}
        for symbol in view.symbols:
            history_df = history.filter(pl.col("symbol") == symbol)
            if history_df.height < self._window:
                continue

            close_prices = [float(v) for v in history_df["adj_close"].to_list()]
            max_price = max(close_prices)
            min_price = min(close_prices)

            breakout_level = max_price if max_price > min_price else min_price
            recent_close = view.latest_close()[symbol]
            volume_series = history_df["volume"]
            entry_date = history_df.filter(pl.col("adj_close") == recent_close)["session_date"].first()
            support_volume = volume_series.filter(
                (pl.col("session_date") >= entry_date) & (pl.col("session_date").lt(stamp))
            ).sum().item()

            if support_volume > 0:
                score = ((recent_close - breakout_level) / breakout_level) * support_volume
                breakout_scores[symbol] = score

        picks: list[str] = []
        for symbol, score in sorted(breakout_scores.items(), key=lambda x: x[1], reverse=True):
            if len(picks) >= self._top_n:
                break
            picks.append(symbol)

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
    newest = visible["session_date"].max().cast(pl.Date)
    assert isinstance(newest, date)
    return newest