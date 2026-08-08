from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "This strategy identifies breakout continuation by leveraging recent price action "
        "and volume trends. It looks for a significant increase in volume on the day of a new "
        "high or low to confirm the breakout and then enters trades aligned with the direction "
        "of the breakout, ensuring higher probability of a sustained trend."
    )

    def __init__(self, lookback_days: int = 5, top_n: int = 10) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty() or history.height < self._lookback_days + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            latest_data = _latest_daily_data(history.filter(pl.col("symbol") == symbol))
            if len(latest_data) < self._lookback_days + 1:
                continue

            high, low = latest_data["high"], latest_data["low"]
            close = float(latest_data[history.height - 1]["close"])
            volume = int(latest_data[history.height - 1]["volume"])

            if close > max(high):
                breakout_high = True
                breakout_low = False
            elif close < min(low):
                breakout_high = False
                breakout_low = True
            else:
                continue

            prev_volume = latest_data.tail(self._lookback_days)["volume"].mean()
            volume_ratio = volume / prev_volume - 1.0

            if volume_ratio > 0.25 and (breakout_high or breakout_low):
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


def _latest_daily_data(frame: pl.DataFrame) -> dict[str, float]:
    last_session = frame.sort("session_date", descending=True).head(1)
    data = {col: float(last_session[col].item()) for col in ["open", "high", "low", "close", "volume"]}
    return data