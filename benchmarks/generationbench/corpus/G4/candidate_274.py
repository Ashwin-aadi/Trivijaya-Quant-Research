from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalReturns(Strategy):
    rationale = (
        "This strategy exploits historical seasonality in the Indian market by identifying "
        "stocks that historically perform well during specific months. By focusing on these "
        "months, we aim to capture alpha through predictable patterns in investor behavior and "
        "market conditions."
    )

    def __init__(self, window: int = 10, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=2 * self._window + 1)

        if history.height < 2 * self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        monthly_returns = (
            history.select(
                pl.col("symbol"),
                pl.col("session_date").dt.month().alias("month"),
                (pl.col("adj_close") / pl.col("adj_close").shift(21) - 1.0).alias("return"),
            )
            .group_by(["symbol", "month"])
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Calculate average returns for each month over the window period
        avg_returns = (
            monthly_returns.group_by("month")
            .select(pl.col("avg_return").mean().alias("monthly_avg_return"))
            .collect()
        )

        # Identify top performing months
        top_months = avg_returns.sort("monthly_avg_return", descending=True).head(self._window)["month"].to_list()

        picks: list[str] = []
        for symbol in view.symbols:
            if (symbol_data := monthly_returns.filter(pl.col("symbol") == symbol)).height < 2 * self._window + 1:
                continue
            # Filter by top performing months and calculate average return during these months
            relevant_returns = symbol_data.filter(pl.col("month").is_in(top_months))
            avg_monthly_return = float(relevant_returns["avg_return"].mean())
            if avg_monthly_return > 0.05:  # Threshold for selecting stocks
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