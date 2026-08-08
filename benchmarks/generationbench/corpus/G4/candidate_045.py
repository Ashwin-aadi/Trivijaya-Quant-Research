from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the empirical phenomenon that lower-volatility stocks tend to outperform "
        "higher-volatility stocks over long periods. By tilting portfolios towards less volatile stocks, "
        "we aim to capitalize on market inefficiencies and risk-averse investor behavior."
    )

    def __init__(self, lookback_days: int = 60, top_n: int = 30) -> None:
        self._lookback_days = lookback_days
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("daily_return")
        ).filter(pl.col("symbol").is_in(view.symbols))

        # Calculate volatility for each stock over the lookback period
        volatilities = (
            history.group_by("symbol")
                   .agg((pl.col("daily_return").std().alias("volatility")))
                   .sort("volatility", descending=False)
                   .head(self._top_n)
        )

        if volatilities.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Rank and assign weights
        weights = {row["symbol"]: 1.0 / len(volatilities) for _, row in volatilities.iter_rows()}
        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest