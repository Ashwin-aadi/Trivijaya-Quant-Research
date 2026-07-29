from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening filters out less liquid stocks to ensure smoother trading "
        "and potentially lower transaction costs. Equal weighting ensures that no single "
        "stock dominates the portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.filter(pl.col("volume").is_not_null())
            .group_by("symbol")
            .agg(
                pl.col("adj_close").mean().alias("avg_price"),
                pl.col("volume").sum().alias("total_volume"),
            )
            .sort("total_volume", descending=True)
            .head(30)  # Select top 30 liquid stocks
        )

        if liquidity_screened.height < 30:
            return Signal(information_available_at=stamp, weights={})

        weights = {symbol: 1.0 / 30 for symbol in liquidity_screened["symbol"].to_list()}
        return Signal(
            information_available_at=stamp,
            weights={s: weights[s] for s in view.symbols if s in weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest