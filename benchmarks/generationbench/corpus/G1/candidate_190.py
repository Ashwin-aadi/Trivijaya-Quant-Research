from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Historical data often show certain stocks in the Indian market exhibiting higher returns "
        "during specific times of the year. This strategy aims to exploit these seasonality effects "
        "by overweighting symbols that have historically performed well during the current month."
    )

    def __init__(self, window: int = 20, lookback_months: int = 12) -> None:
        self._window = window
        self._lookback_months = lookback_months

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Get the current month
        current_month = stamp.month

        # Filter the data for the last `lookback_months` months
        history = view.history()
        filtered_history = (
            history.filter(pl.col("session_date").dt.month().is_in(
                range(current_month - self._lookback_months + 1, current_month + 1)
            ))
        )

        if filtered_history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the average monthly return for each symbol
        avg_returns = (
            filtered_history.group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("avg_return")
            )
            .sort("avg_return", descending=True)
            .select(["symbol", "avg_return"])
        )

        # Get the top symbols with highest average returns
        top_symbols = avg_returns.head(self._lookback_months)["symbol"].to_list()

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest