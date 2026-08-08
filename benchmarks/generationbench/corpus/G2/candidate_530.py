from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High volatility indicates that a security is experiencing significant price action, "
        "potentially due to market interest or news events. Trend following strategies aim to "
        "capitalize on these movements by identifying securities with both high volatility and "
        "positive trends."
    )

    def __init__(self, window: int = 50, threshold: float = 1.5) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = closes.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
        )

        # Filter out null values and compute mean return over the window
        non_null_returns = returns.drop_nulls(subset="r")
        avg_return = (
            non_null_returns.groupby("symbol").agg(
                (pl.col("r").mean()).alias("avg_return")
            ).select("avg_return").to_series().to_list()
        )

        # Calculate volatility as the standard deviation of daily returns
        volatilities = [
            float(v) for v in
            non_null_returns.group_by("symbol").agg(
                (pl.col("r").std()).alias("volatility")
            ).select("volatility").to_series().to_list()
        ]

        # Identify symbols with both high volatility and positive average return
        picks: list[str] = []
        for symbol, avg_ret, vol in zip(non_null_returns["symbol"].to_list(), avg_return, volatilities):
            if avg_ret >= 0.0 and vol > self._threshold:
                picks.append(symbol)

        # Compute equal weights if we have identified any symbols
        weight = 1.0 / len(picks) if picks else 0.0
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