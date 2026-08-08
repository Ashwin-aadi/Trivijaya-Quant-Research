from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets often reflects predictable patterns related to "
        "calendar events or weather. This strategy exploits historical trends by buying "
        "stocks that have historically outperformed during specific months."
    )

    def __init__(self, window: int = 60, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        monthly_returns = {}
        for symbol in view.symbols:
            closes = history.select(
                pl.col("session_date").dt.month().alias("month"),
                pl.col("adj_close").alias("close"),
            )
            month_grouped = closes.groupby("month").agg(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return")
            )

            monthly_returns[symbol] = (
                month_grouped.select(pl.sum("return")).item()
                if not month_grouped.is_empty()
                else 0.0
            )

        top_performing_symbols = sorted(
            monthly_returns.items(), key=lambda x: x[1], reverse=True
        )[: self._top_n]

        weights = {symbol: weight for symbol, _ in top_performing_symbols if _ > 0}
        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items() if w > 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest