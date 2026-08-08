from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality effects suggest that certain times of the year are more favorable for "
        "certain stocks due to calendar events such as fiscal year ends, regulatory changes, "
        "or investor sentiment. Identifying these trends can provide trading opportunities."
    )

    def __init__(self, window: int = 90) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_effects: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history = view.history().select(
                pl.col("session_date").dt.month(), pl.col(symbol)
            )
            monthly_closes = history.groupby("session_date.dt.month()").agg(
                (pl.col(symbol).mean()).alias("monthly_close")
            ).sort("session_date.dt.month()")

            if monthly_closes.height >= 12:
                seasonal_trend = (
                    monthly_closes.select(pl.all().tail(6))
                    .select((pl.col("monthly_close") - pl.col("monthly_close").shift(1)).mean())
                    .get("monthly_close")
                    .item()
                )
                if seasonal_trend > 0.01:
                    seasonal_effects[symbol] = seasonal_trend

        top_trends: list[str] = sorted(seasonal_effects, key=seasonal_effects.get, reverse=True)[:5]
        if not top_trends:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_trends)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_trends},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest