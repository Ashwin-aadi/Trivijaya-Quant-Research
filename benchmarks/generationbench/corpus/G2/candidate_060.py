from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in recent periods to continue outperforming. This strategy "
        "buys top performers and sells laggards."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        rank_df = (
            history
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window - 1) - 1.0).alias("return"),
            )
            .sort("return", descending=True)
            .select(["symbol", "return"])
            .head(5)
        )

        top_symbols = rank_df["symbol"].to_list()
        bottom_symbols = rank_df.sort("return").tail(5)["symbol"].to_list()

        if not top_symbols or not bottom_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_top = 0.6 / len(top_symbols)
        weight_bottom = -0.4 / len(bottom_symbols)

        weights = {s: weight_top for s in top_symbols}
        for s in bottom_symbols:
            weights[s] = weight_bottom

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest