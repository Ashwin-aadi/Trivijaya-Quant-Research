from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "High-volatility stocks are more likely to continue their recent trend due to the "
        "greater momentum of their price movements. By scaling our exposure to these high-volatility "
        "stocks, we can capture this momentum effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.sort("session_date")
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .select(pl.col("symbol"), "return")
            .collect()
        )

        # Calculate volatility
        vol = (
            returns.groupby("symbol")
            .agg((pl.col("return").std().alias("volatility")))
            .sort("volatility", descending=True)
            .select("symbol", "volatility")
            .to_pandas()["volatility"].to_list()
        )

        # Scale by volatility
        vol_scaled_returns = returns.join(
            pl.DataFrame({"symbol": returns["symbol"], "volatility": vol}),
            on="symbol",
        )
        scaled_returns = (
            vol_scaled_returns.with_columns(
                (pl.col("return") * pl.col("volatility")).alias("scaled_return")
            )
            .select(pl.col("symbol"), "scaled_return")
            .groupby("symbol")
            .agg((pl.col("scaled_return").mean().alias("mean_return")))
            .sort("mean_return", descending=True)
            .select("symbol", "mean_return")
            .to_pandas()
        )

        # Select top symbols based on mean return
        picks = scaled_returns["symbol"].tolist()[:5]
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