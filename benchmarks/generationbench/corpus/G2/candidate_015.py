from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for the efficiency of the market. More liquid stocks are likely to "
        "have tighter bid-ask spreads and less price volatility. By equal weighting these more "
        "liquid stocks, we aim to benefit from their lower transaction costs and potentially "
        "higher liquidity premiums."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"),
                pl.col("session_date").max().alias("last_traded_date"),
            )
            .sort("liquidity_score", descending=True)
            .filter(pl.col("last_traded_date") == view.as_of - date(self._window, 0, 0))
            .head(10)  # Select top 10 by liquidity score
        )

        if liquidity_screened.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = liquidity_screened["symbol"].to_list()
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest