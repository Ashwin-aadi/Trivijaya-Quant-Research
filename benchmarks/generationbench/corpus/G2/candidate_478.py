from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalBreakout(Strategy):
    rationale = (
        "Seasonality effects can provide predictable patterns in stock prices due to "
        "recurring events or human behavior. For instance, certain sectors might perform"
        " better during specific times of the year. This strategy aims to exploit such "
        "patterns by identifying breakout opportunities from seasonal lows."
    )

    def __init__(self, window: int = 30, breakout_threshold: float = 1.05) -> None:
        self._window = window
        self._breakout_threshold = breakout_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 365 * 2)
        if history.height < self._window + 365 * 2:
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(
            pl.col("session_date").alias("date"),
            pl.col("adj_close").alias("close"),
        )

        # Filter by year to isolate seasonal effects
        for year in [view.as_of.year - 1, view.as_of.year]:
            mask = (closes["date"].dt.year() == year) & (
                closes["date"].dt.month() == view.as_of.month()
            )
            yearly_closes = closes.filter(mask)
            if not yearly_closes.height:
                continue

            # Identify seasonal low
            lowest_close = min(yearly_closes["close"])
            breakout_high = (yearly_closes["close"] >= lowest_close * self._breakout_threshold).sum()

            if breakout_high > 0.5 * len(yearly_closes):
                picks: list[str] = []
                for symbol in view.symbols:
                    if not history.select(pl.col("symbol") == symbol).is_empty():
                        pick_history = history.filter(
                            (pl.col("symbol") == symbol) & mask
                        ).select(["close"])
                        if (
                            pick_history.height >= self._window
                            and pick_history["close"].to_list()[-1] > max(pick_history["close"].to_list())
                        ):
                            picks.append(symbol)

                if picks:
                    weight = 1.0 / len(picks)
                    return Signal(
                        information_available_at=stamp,
                        weights={s: weight for s in picks},
                    )

        return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest