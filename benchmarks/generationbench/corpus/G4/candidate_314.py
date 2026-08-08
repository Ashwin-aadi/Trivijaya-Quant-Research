from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits mean-reverting behavior within volatile regimes by "
        "scaling position sizes based on 20-day rolling standard deviation of daily returns. "
        "During high volatility, it reduces position size to maintain risk control, while "
        "increasing exposure during low-volatility periods."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Compute rolling standard deviation of returns
        volatilities = (
            history.group_by("symbol")
            .agg((pl.col("return").std().over(pl.col("session_date").shift(-self._window + 1)).alias(f"volatility_{self._window}"))
                 )
            .select([pl.col("symbol"), f"volatility_{self._window}"])
        )

        # Filter out symbols without enough data
        if volatilities.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Rank by volatility for both high and low volatility scenarios
        ranking_high_vol = volatilities.sort(f"volatility_{self._window}", descending=False).select(pl.col("symbol"))
        ranking_low_vol = volatilities.sort(f"volatility_{self._window}", descending=True).select(pl.col("symbol"))

        # Determine current market view (high or low volatility)
        mean_volatility = history.select([pl.col("return").std().alias("mean_volatility")])["mean_volatility"].item()
        is_high_volatility = history["return"].std().item() > 1.5 * mean_volatility

        if is_high_volatility:
            picks = ranking_low_vol.head(self._top_n).to_dict(False)
        else:
            picks = ranking_high_vol.head(self._top_n).to_dict(False)

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