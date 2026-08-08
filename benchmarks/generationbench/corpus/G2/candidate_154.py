from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in stock markets can arise due to recurring events such as festivals, "
        "earnings seasons, or regulatory changes. By identifying stocks that show strong "
        "historical performance during specific months of the year, we can exploit these trends."
    )

    def __init__(self, lookback_years: int = 3, threshold: float = 0.15) -> None:
        self._lookback_years = lookback_years
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_years * 252)
        if history.height < 252 * self._lookback_years:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average monthly returns for each stock
        avg_returns: dict[str, float] = {}
        symbols = view.symbols
        for symbol in symbols:
            monthly_returns = history.select(
                pl.col("session_date").dt.month().alias("month"),
                pl.col(symbol).alias("adj_close")
            ).group_by("month").agg((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("avg_return"))

            if not monthly_returns.is_empty():
                avg_return = float(monthly_returns.select(pl.col("avg_return")).item())
                if avg_return > self._threshold:
                    avg_returns[symbol] = avg_return

        # Select the top performing symbols
        top_symbols = sorted(avg_returns.keys(), key=lambda x: avg_returns[x], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest