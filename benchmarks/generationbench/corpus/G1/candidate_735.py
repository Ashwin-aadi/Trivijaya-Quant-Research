from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for tradability and market efficiency. "
        "By equal-weighting the most liquid stocks, we aim to capture more trading opportunities. "
        "The strategy focuses on recent liquidity measures to ensure that the chosen stocks are active in the market."
    )

    def __init__(self, lookback_days: int = 20) -> None:
        self._lookback_days = lookback_days

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_days)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity = (
            history.filter(pl.col("session_date") >= (stamp - pl.duration(days=self._lookback_days)))
            .group_by("symbol")
            .agg(
                volume_sum=pl.col("volume").sum(),
                trade_count=pl.col("close").count(),
            )
            .with_columns(
                liquidity_score=(pl.col("volume_sum") / pl.col("trade_count")).alias("liquidity_score"),
            )
        )

        if liquidity.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = liquidity.sort("liquidity_score", descending=True)["symbol"].to_list()[:5]
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