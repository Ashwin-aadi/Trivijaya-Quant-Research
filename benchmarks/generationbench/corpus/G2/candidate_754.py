from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Seasonal effects can arise from various factors such as corporate earnings reports, "
        "government policies, or cultural events. These periodic patterns can lead to predictable "
        "price movements that can be exploited through quantitative strategies."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(
            pl.col("session_date").cast(pl.Date),
            pl.col("symbol"),
            pl.col("adj_close")
        ).with_columns(
            (pl.col("session_date") - pl.col("session_date").shift(self._window)).alias("days_diff")
        )

        seasonality_map: dict[date, list[str]] = {}
        for row in closes.rows():
            date_ = row["session_date"]
            symbol = row["symbol"]
            if not isinstance(date_, date):
                continue
            day_of_year = date_.timetuple().tm_yday

            if day_of_year not in seasonality_map:
                seasonality_map[day_of_year] = []

            seasonality_map[day_of_year].append(symbol)

        top_days: list[date] = sorted(seasonality_map.keys(), key=lambda x: len(seasonality_map[x]), reverse=True)[:self._top_n]

        picks: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            for day_of_year in top_days:
                date_ = pl.date(year=date.today().year, month=1, day=day_of_year).cast(pl.Date)
                if len(values) >= self._window and values[-self._window] > max(values[:-self._window]):
                    picks[symbol] = 1.0 / len(top_days)

        return Signal(
            information_available_at=stamp,
            weights=picks
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date).to_python_data()
    assert isinstance(newest, date)
    return newest