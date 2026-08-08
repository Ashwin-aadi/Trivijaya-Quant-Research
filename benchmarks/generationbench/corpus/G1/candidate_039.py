from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and investor interest. By selecting assets with higher "
        "liquidity, the strategy aims to reduce transaction costs and ensure that positions can be entered or exited "
        "without significant impact on the asset's price."
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
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .sort("avg_volume", descending=True)
            .head(self._window)
        )

        if liquidity_screened.height < self._window:
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