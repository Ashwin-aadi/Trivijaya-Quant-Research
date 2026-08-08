from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Seasonality in stock markets can arise from a variety of factors, such as earnings "
        "announcements, consumer behavior patterns, or regulatory calendars. By exploiting "
        "known seasonal patterns, one might be able to identify stocks that perform well at "
        "certain times of the year."
    )

    def __init__(self, window: int = 365, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Identify symbols with significant seasonal patterns
        seasonality_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            daily_closes = history.select(
                pl.col("session_date"), pl.col(symbol).alias("close")
            ).with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return")
            )

            # Calculate the average return over the year
            avg_return = daily_closes.select(
                pl.col("return").mean().alias("avg_return")
            ).collect()["avg_return"][0]

            if abs(avg_return) >= self._threshold:
                seasonality_scores[symbol] = avg_return

        top_symbols = sorted(seasonality_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        weights = {symbol: 0.2 for symbol, _ in top_symbols}

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest