from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects can arise from various economic and psychological factors. For "
        "example, certain stocks may exhibit higher returns during specific times of the year "
        "due to increased trading volumes or market expectations. By identifying these patterns, "
        "we can exploit them for profit."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol = history["symbol"][0]
        closes = history.select(pl.col("adj_close"))

        # Calculate the average close and the seasonally adjusted close
        avg_closes = closes.mean().item()
        seasonality_factor = (
            closes.with_columns(
                (pl.col("adj_close") / avg_closes).alias("seasonal_factor")
            )
            .sort("session_date", descending=False)
            .select(pl.col("seasonal_factor"))
            .to_list()[0]
        )

        # Identify the season in which we are currently
        today = stamp.toordinal()
        start_of_year = date(stamp.year, 1, 1).toordinal()
        days_in_year = (date(stamp.year + 1, 1, 1) - date(stamp.year, 1, 1)).days

        season_start = max(0, int((today - start_of_year) / days_in_year * self._window))
        season_end = min(self._window - 1, season_start + (self._window // 4))

        # Compute the seasonal trend
        seasonal_trend = (
            history.slice(season_start, season_end)
            .select(pl.col("adj_close"))
            .mean()
            .item()
        )

        if len(view.symbols) == 0 or symbol not in view.symbols:
            return Signal(information_available_at=stamp, weights={})

        # Generate signal based on the seasonal trend
        weight = 1.0 / (len(view.symbols))
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in view.symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest