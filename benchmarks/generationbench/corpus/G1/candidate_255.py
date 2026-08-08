from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key indicator of marketability. By selecting the most liquid stocks, "
        "we aim to minimize transaction costs and execution risks, ensuring smoother trading."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"),
                pl.col("session_date").max().alias("last_trade_date"),
            )
            .filter(pl.col("volume").sum() > 0)  # Ensure volume is not zero
        )

        symbols = liquidity_screened.select("symbol").to_list()[0]
        if len(symbols) < 5:
            return Signal(information_available_at=stamp, weights={})

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