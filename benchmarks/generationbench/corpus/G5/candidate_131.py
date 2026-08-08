from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "This strategy combines two weakly related characteristics: the 20-day simple moving "
        "average of daily returns and the number of days a stock has been in an uptrend over "
        "the last month. Stocks with high values for both metrics are considered strong."
    )

    def __init__(self, ma_window: int = 20, trend_days: int = 30) -> None:
        self._ma_window = ma_window
        self._trend_days = trend_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._trend_days + 1)
        if closes.height < self._trend_days + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple moving average of daily returns
        price_changes = (closes["adj_close"] / closes["adj_close"].shift(1) - 1.0).alias("r")
        ma_series = (
            view.history().with_columns(price_changes)
            .sort("session_date")
            .group_by("symbol")
            .agg((pl.col("r").mean()).alias("ma"))
        )

        # Identify uptrend days
        uptrend_days = []
        for symbol in view.symbols:
            if symbol not in closes.columns or len(closes[symbol].drop_nulls()) < self._trend_days + 1:
                continue
            close_series = [float(v) for v in closes[symbol].to_list()]
            trend = all(close_series[i] >= close_series[i - 1] for i in range(1, self._trend_days))
            uptrend_days.append((symbol, trend))

        # Filter and rank based on combined metric
        scores: list[tuple[float, str]] = []
        for symbol, trend in uptrend_days:
            ma_value = float(ma_series.filter(pl.col("symbol") == symbol)["ma"].item())
            score = (ma_value + 1 if trend else ma_value)
            scores.append((score, symbol))

        # Select top stocks
        top_scores = sorted(scores, reverse=True)[:5]
        picks = [t[1] for t in top_scores]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest