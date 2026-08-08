from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "High-volatility stocks are more likely to continue their recent trends due to "
        "momentum effects. By investing in the most volatile stocks, we aim to capture "
        "these trending opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .sort("session_date", descending=False)
            .drop_nulls(subset=["returns"])
        )

        # Calculate volatility
        volatilities = (
            history.groupby("symbol")
            .agg(
                (pl.col("returns").std().alias(f"volatility_{self._window}"))
            )
            .sort(pl.col(f"volatility_{self._window}"), descending=True)
            .head(5)
        )

        if volatilities.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Select top 3 most volatile symbols
        picks = [row["symbol"] for row in volatilities.to_dicts()[:3]]

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