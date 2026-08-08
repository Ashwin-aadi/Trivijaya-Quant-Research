from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityStrategy(Strategy):
    rationale = (
        "This strategy exploits historical seasonality in India's equity market by "
        "buying stocks during underperforming months and selling during overperforming ones."
    )

    def __init__(self, lookback_years: int = 10) -> None:
        self._lookback_years = lookback_years

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=365 * self._lookback_years)
        if closes.height < 365 * self._lookback_years:
            return Signal(information_available_at=stamp, weights={})

        # Calculate average returns by month
        avg_returns = _calculate_monthly_avg_returns(closes)

        # Identify underperforming and overperforming months
        top_months = avg_returns.sort("avg_return", descending=False).head(5)
        bottom_months = avg_returns.sort("avg_return", descending=True).head(5)

        # Determine the current month
        current_month = stamp.month

        # Define rules for buying/selling
        if current_month in [m for m, _ in top_months.iter_rows()]:
            signal_weights = {s: 1.0 / len(top_months) for s in view.symbols}
        elif current_month in [m for m, _ in bottom_months.iter_rows()]:
            signal_weights = {}
        else:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp, weights=signal_weights
        )


def _calculate_monthly_avg_returns(closes: pl.DataFrame) -> pl.DataFrame:
    symbols = closes.columns[1:]
    avg_returns = (
        closes.group_by("session_date")
              .agg((pl.col(symbols).shift(0) / pl.col(symbols).shift(365) - 1.0)
                   .mean().alias("avg_return"))
              .with_columns(pl.col("session_date").dt.month_name().alias("month"))
    )
    return avg_returns.groupby("month").agg(
        (pl.col("avg_return").mean()).alias("avg_return")
    ).sort("avg_return", descending=True)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest