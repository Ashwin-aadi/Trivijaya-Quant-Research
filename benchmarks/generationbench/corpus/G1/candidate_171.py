from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and information efficiency. "
        "Highly liquid stocks are less likely to be mispriced, providing a more reliable base for investment decisions."
    )

    def __init__(self, liquidity_window: int = 20) -> None:
        self._liquidity_window = liquidity_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._liquidity_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history.columns]
        liquidity_scores = (
            history.filter(pl.col("session_date") >= (stamp - self._liquidity_window).isoformat())
                .group_by("symbol")
                .agg(
                    pl.col("volume").sum().alias("total_volume"),
                    (pl.col("adj_close").max() - pl.col("adj_close").min()).alias("price_range"),
                )
                .with_columns(
                    (pl.col("total_volume") / pl.col("price_range")).alias("liquidity_score")
                )
        )

        if liquidity_scores.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = [s[0] for s in liquidity_scores.sort("liquidity_score", descending=True).rows()]
        top_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest