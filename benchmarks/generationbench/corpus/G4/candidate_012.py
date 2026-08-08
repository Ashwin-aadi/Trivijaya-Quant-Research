from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum6m(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by selecting stocks with strong "
        "recent price performance relative to their peers. It aims to capitalize on the "
        "persistence of stock price momentum."
    )

    def __init__(self, lookback_days: int = 120) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate simple returns
        history = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_days) - 1.0).alias("return")
            )
            .drop_nulls(["symbol"])
            .sort("session_date", descending=True)
        )

        # Rank symbols by return in descending order
        ranked = history.group_by("symbol").agg(
            pl.col("return").mean().alias("avg_return")
        ).sort("avg_return", descending=True)

        top_n = min(ranked.height, 30)  # Select the top N stocks with highest momentum
        picks: list[str] = [row["symbol"] for row in ranked.to_dicts()[:top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        long_portfolio = {s: weight for s in picks}

        # Short the bottom N stocks
        bottom_n = min(ranked.height, 30)  # Select the bottom N stocks with lowest momentum
        short_picks: list[str] = [row["symbol"] for row in ranked.to_dicts()[bottom_n:]]
        short_weight = -1.0 / len(short_picks)
        short_portfolio = {s: short_weight for s in short_picks}

        weights = {**long_portfolio, **short_portfolio}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest