from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonal trends in equity markets can be exploited by identifying stocks that "
        "tend to perform better during certain times of the year. This strategy aims to "
        "capitalize on historical patterns to generate buy signals."
    )

    def __init__(self, window: int = 365, seasonal_periods: tuple[int, ...] = (30, 91)) -> None:
        self._window = window
        self._seasonal_periods = seasonal_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol = "NIFTY100"  # Assuming the strategy is applied to NIFTY100 constituents
        if symbol not in history.columns:
            return Signal(information_available_at=stamp, weights={})

        daily_returns = _calculate_daily_returns(history)
        seasonal_trends = _extract_seasonal_trends(daily_returns)

        if len(seasonal_trends) < max(self._seasonal_periods):
            return Signal(information_available_at=stamp, weights={})

        recent_trend = max(seasonal_trends[-self._seasonal_periods[0] :])
        if recent_trend > 0:
            weight = 1.0 / len(history["symbol"].unique().to_list())
            selected_symbols = history.select(["symbol"]).unique().to_dict(False)
            return Signal(
                information_available_at=stamp,
                weights={s: weight for s in selected_symbols},
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
    )
    return df.sort("session_date")


def _extract_seasonal_trends(df: pl.DataFrame) -> list[float]:
    trends = []
    for period in [30, 91]:  # Example seasonal periods
        grouped = df.groupby("symbol").agg(
            (pl.col("daily_return").mean().alias(f"mean_return_{period}"))
        )
        mean_returns = grouped.select([f"mean_return_{period}"])
        trends.extend(mean_returns.to_dict(False)[0].values())
    return trends