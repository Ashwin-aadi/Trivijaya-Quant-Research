from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data often shows that certain stocks exhibit higher returns during specific "
        "times of the year due to seasonal effects. This strategy aims to capture these gains by "
        "allocating capital to sectors or individual stocks that have historically performed well "
        "during a particular month."
    )

    def __init__(self, season: int = 12) -> None:
        self._season = season

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=365 * 4)  # Use a lookback of 4 years
        if history.height < 1200:  # At least 10 years of data is needed for reliable seasonality
            return Signal(information_available_at=stamp, weights={})

        grouped = history.group_by("symbol").agg(
            (pl.col("session_date").dt.month().alias("month")),
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
        )
        means = grouped.select(
            [pl.col("symbol"), pl.col("month").mean(), pl.col("r").mean()]
        ).collect()

        top_symbols: list[str] = []
        for _, row in means.iter_rows():
            if row["month"].item() == self._season and abs(row["r"].item()) > 0.1:
                top_symbols.append(row["symbol"].item())

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()  # Use item() to get the date object
    assert isinstance(newest, date)
    return newest