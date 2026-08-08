from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Historically, lower volatility stocks tend to outperform higher volatility ones over time "
        "due to risk aversion among investors. This strategy selects the top decile of least volatile "
        "stocks based on their 12-month historical volatility scores."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                ((pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r"))
            )
            .sort("session_date")
            .drop_nulls()
        )

        # Compute 12-month volatility for each stock
        volatilities = (
            history.groupby("symbol")
            .agg((pl.col("r").std().alias("volatility")))
            .select(
                pl.col("symbol"),
                (pl.col("volatility") * 100).alias("volatility_pct"),  # Convert to percentage for readability
            )
        )

        # Rank stocks by volatility
        ranks = volatilities.sort("volatility", descending=False).with_column(
            pl.arange(1, pl.count() + 1).over("symbol").alias("rank")
        )

        # Get top decile of low-volatility stocks
        top_decile_count = int(volatilities.height * 0.1)
        selected_symbols = ranks.filter(pl.col("rank") <= top_decile_count)["symbol"].to_list()

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Equal weight allocation for the selected stocks
        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in selected_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest