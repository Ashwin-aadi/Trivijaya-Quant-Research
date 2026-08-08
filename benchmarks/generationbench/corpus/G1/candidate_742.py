from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in the stock market refers to recurring patterns related to seasonal "
        "factors. Historical data often reveal that certain stocks perform better during specific times of the year."
    )

    def __init__(self, window: int = 365, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            # Calculate the average close price per month and find the months with highest and lowest prices
            monthly_closes = view.closes(lookback=self._window).group_by(
                pl.date_range(view.as_of, stamp, duration="1mo", closed='left')
            ).agg(pl.col("adj_close").mean().alias("monthly_avg"))
            top_months = [date(year=row["year"], month=row["month"], day=1) for row in monthly_closes.sort("monthly_avg", descending=True).head(self._top_n)]
            # Check if the last close date falls within the top months
            if any(date(row["year"], row["month"], 1) in top_months for row in pl.date_range(view.as_of, stamp, duration="1d").to_dict()):
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