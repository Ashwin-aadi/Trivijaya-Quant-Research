from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a proxy for marketability and can indicate the ease of trading. "
        "This strategy screens for symbols with high liquidity and then equally weights "
        "them to exploit potential market inefficiencies."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_symbols = (
            history.group_by("symbol")
            .agg(
                pl.col("volume").sum().alias("total_volume"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean()
                * 252
                .alias("average_return"),
            )
            .sort("total_volume", descending=True)
            .select(["symbol"])
            .head(self._window)["symbol"]
        )

        if liquidity_screened_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        equal_weight = 1.0 / len(liquidity_screened_symbols.to_list())
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: equal_weight for symbol in liquidity_screened_symbols
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest