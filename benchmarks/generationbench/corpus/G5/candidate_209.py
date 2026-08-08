from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a strong breakout in one direction, the price is likely to continue moving in that "
        "direction due to inertia. This strategy identifies stocks where recent prices have moved beyond their recent range and continues to move in that direction."
    )

    def __init__(self, window: int = 20, continuation_threshold: float = 1.5) -> None:
        self._window = window
        self._continuation_threshold = continuation_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history_df = view.history(lookback=self._window)
        if history_df.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history_df["symbol"]:
                continue

            df = history_df.filter(pl.col("symbol") == symbol)
            high_prices = [float(v) for v in df["high"].to_list()]
            low_prices = [float(v) for v in df["low"].to_list()]

            if len(high_prices) < self._window or len(low_prices) < self._window:
                continue

            recent_high = max(high_prices[-self._window:])
            recent_low = min(low_prices[-self._window:])
            breakout_price = max(df["adj_close"][-1], df["close"][-1])
            continuation_price = recent_high + (recent_high - recent_low) * self._continuation_threshold

            if continues_in_direction(breakout_price, high_prices):
                picks.append(symbol)
            elif continues_in_direction(breakout_price, low_prices):
                picks.append(symbol)

        picks = list(set(picks))[: self._window]
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


def continues_in_direction(breakout_price: float, prices: list[float]) -> bool:
    for price in prices[::-1]:
        if price < breakout_price:
            return False
    return True