from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are often considered less risky and may have more market "
        "impartiality. By equal-weighting these stocks, we aim to capture the performance of "
        "the most liquid equities in the NIFTY 100."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.select(pl.col("symbol"))
            .group_by("symbol")
            .agg(
                pl.col("volume").mean().alias("avg_volume"),
                (pl.col("adj_close") / pl.col("open")).mean().alias("price_range"),
            )
            .filter((pl.col("avg_volume") > 10_000) & (pl.col("price_range") < 2.0))
        )

        if liquidity_screened.height == 0:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in liquidity_screened["symbol"].to_list()]
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