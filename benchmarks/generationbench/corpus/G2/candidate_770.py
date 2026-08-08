from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend-following strategies exploit the idea that assets with higher "
        "recent volatility are more likely to continue their recent direction. This is based on "
        "the assumption that high volatility often precedes a continuation of the trend, as market "
        "participants adjust their positions."
    )

    def __init__(self, window: int = 20, scaling_factor: float = 1.0) -> None:
        self._window = window
        self._scaling_factor = scaling_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
        )

        # Filter out null values for the last day's return
        history = history.drop_nulls(subset=["return"])

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate volatility as standard deviation of returns over the window period
        volatility = (history.select(pl.col("return").std()) * self._scaling_factor).item()

        # Identify symbols with high volatility
        top_symbols = (
            history.groupby("symbol")
            .agg(
                pl.col("return").mean().alias("avg_return"),
                pl.col("return").std().alias("volatility"),
            )
            .sort(pl.col("volatility"), descending=True)
            .head(self._window // 2)["symbol"]
        )

        # Select top symbols based on volatility
        picks: list[str] = [s for s in top_symbols.to_list() if s in history.columns]

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