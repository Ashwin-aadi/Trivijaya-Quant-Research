from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting allocates capital based on the liquidity of "
        "individual stocks. This ensures that more liquid stocks receive a higher weight in "
        "the portfolio, potentially reducing trading costs and improving market impact."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns"),
            )
            .with_columns(
                (pl.col("total_volume") / history["volume"].sum()).alias("liquidity_score")
            )
        )

        if liquidity_scores.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = (
            liquidity_scores.sort("liquidity_score", descending=True)
            .select(["symbol"])
            .to_series()
            .to_list()[:5]
        )

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest