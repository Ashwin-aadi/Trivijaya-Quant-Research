from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuation breakout strategies identify stocks that have just formed a "
        "breakout and predict they will continue to trend. By buying such stocks, we aim "
        "to capture the momentum of the breakout."
    )

    def __init__(self, window: int = 20, lookback_days: int = 10) -> None:
        self._window = window
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback_days)
        if history.is_empty() or history.height < self._window + self._lookback_days:
            return Signal(information_available_at=stamp, weights={})

        breakout_sigs: list[str] = []
        for symbol in view.symbols:
            hist_df = history.select(
                pl.col("session_date"), pl.col(symbol).alias("adj_close")
            )
            if hist_df.height < self._window + self._lookback_days:
                continue

            # Find the breakout day
            breakout_day: date | None = _find_breakout_day(hist_df, self._window)
            if not breakout_day:
                continue

            # Check continuation condition for at least self._lookback_days after breakout
            if hist_df.filter(pl.col("session_date") > breakout_day).height < self._lookback_days + 1:
                continue

            breakout_sigs.append(symbol)

        weight = 1.0 / len(breakout_sigs)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_sigs}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_breakout_day(df: pl.DataFrame, window: int) -> date | None:
    breakout_dates: list[date] = []
    for i in range(window, df.height):
        current_close = float(df.select(pl.col("adj_close").at(i - 1)).item())
        previous_max = float(df.select(pl.col("adj_close").slice(0, window).max().at(0)).item())
        if current_close > previous_max:
            breakout_dates.append(df.select(pl.col("session_date").at(i - 1)).item())

    if not breakout_dates:
        return None

    # Find the most recent breakout day
    latest_breakout = max(breakout_dates)
    return latest_breakout