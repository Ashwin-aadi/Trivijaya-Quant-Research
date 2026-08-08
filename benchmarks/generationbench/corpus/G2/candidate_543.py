from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Higher liquidity stocks are often considered less risky and can provide more stable returns. "
        "By equal-weighting these highly liquid stocks, the strategy aims to capture the benefits of reduced risk while ensuring diversification."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Screen for liquidity
        liquidity_screened = history.group_by("symbol").agg(
            (pl.col("volume").mean().alias("avg_volume"))
        ).sort("avg_volume", descending=True).select(["symbol", "avg_volume"])

        top_n_symbols = liquidity_screened.head(self._window)["symbol"].to_list()

        if len(top_n_symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(top_n_symbols, [weight] * len(top_n_symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest