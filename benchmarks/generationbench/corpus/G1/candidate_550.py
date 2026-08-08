from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Equity markets exhibit seasonality effects where certain times of the year "
        "are associated with higher returns. This strategy identifies such periods and allocates "
        "capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Extract the month from the session_date column
        history = history.with_columns(pl.col("session_date").dt.month().alias("month"))

        # Calculate mean returns by month
        monthly_returns = (
            history.group_by("month")
            .agg(
                (pl.col("close") / pl.col("close").shift(self._window) - 1.0).alias("return")
            )
            .select(["month", "return"])
        )

        # Find the best performing month
        best_month = monthly_returns.sort("return", descending=True)["month"][0]

        # Filter stocks in the best performing month
        symbols_in_best_month = (
            history.filter(pl.col("month") == best_month).select("symbol").unique().to_dict()["symbol"]
        )

        if not symbols_in_best_month:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(symbols_in_best_month)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols_in_best_month},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest