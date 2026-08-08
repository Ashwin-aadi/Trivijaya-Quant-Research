from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for assets that have outperformed "
        "in the recent past to continue outperforming. This phenomenon is often attributed to "
        "market sentiment and the reluctance of traders to sell winners too quickly."
    )

    def __init__(self, lookback: int = 60, top_n: int = 10) -> None:
        self._lookback = lookback
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") - pl.col("adj_close").shift(self._lookback)) / pl.col("adj_close").shift(self._lookback).alias("momentum_score"),
            )
            .sort("momentum_score", descending=True)
            .head(self._top_n)[["symbol"]]
        )

        weights = {s: 1.0 / len(momentum_scores) for s in momentum_scores["symbol"].to_list()}
        return Signal(
            information_available_at=stamp, weights={k: float(v) for k, v in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest