from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighing(Strategy):
    rationale = (
        "Higher liquidity stocks are often more attractive to institutional investors and "
        "traders due to lower transaction costs. Therefore, a strategy that weights towards "
        "highly liquid stocks may benefit from reduced execution costs and potentially better "
        "price impact."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history
            .group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") - pl.col("open")).abs().mean().alias("price_volatility"),
            )
        )

        # Calculate a score based on the sum of volume and price volatility
        liquidity_scores = (
            liquidity_scores
            .with_columns(
                ((pl.col("total_volume") + 1) * (10 - pl.col("price_volatility"))).alias("liquidity_score")
            )
            .sort("liquidity_score", descending=True)
        )

        top_symbols = [row["symbol"] for row in liquidity_scores.to_dicts()[: self._window]]

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