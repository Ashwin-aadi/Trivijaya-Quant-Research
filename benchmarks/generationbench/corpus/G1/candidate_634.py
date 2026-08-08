from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data from the Indian stock market suggests that certain months of the year "
        "show higher returns due to specific seasonal effects or calendar events. This strategy "
        "exploits these historical patterns by overweighting stocks in periods known to perform well."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Filter by month
        month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                     7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        history = history.with_column(pl.col("session_date").str.strptime(pl.Date, fmt="%Y-%m-%d")
                                      .alias("parsed_date"))
        history = history.with_column((pl.col("parsed_date").dt.month()).cast(pl.Int32).alias("month"))

        # Identify high-performing months
        monthly_avg_returns: dict[str, float] = {}
        for month in range(1, 13):
            month_filter = (history["month"] == month)
            month_grouped = history.filter(month_filter).group_by("symbol").agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
            )
            if not month_grouped.is_empty():
                avg_return = float(month_grouped.select(pl.col("return").mean()).item())
                monthly_avg_returns[month_map[month]] = avg_return

        # Sort by average return
        sorted_months = [k for k, v in sorted(monthly_avg_returns.items(), key=lambda item: item[1], reverse=True)]
        picks: list[str] = []
        for month in sorted_months[:5]:
            symbols_in_month = history.filter((history["month"] == month_map_to_num[month]) & month_filter)["symbol"].to_list()
            picks.extend(symbols_in_month)

        # Select top-performing symbols
        picks = list(set(picks))[:10]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().date()
    assert isinstance(newest, date)
    return newest

# Map month names back to numbers for filtering
month_map_to_num = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
                    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}