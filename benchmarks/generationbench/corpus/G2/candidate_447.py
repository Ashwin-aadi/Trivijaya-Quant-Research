from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are more likely to be continuously priced and less susceptible "
        "to sudden price jumps. By equal-weighting high-liquidity stocks, the strategy aims to "
        "benefit from consistent pricing while reducing the risk of adverse price movements."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").mean() / pl.col("volume").std()).alias("liquidity_score")
            )
            .sort("liquidity_score", descending=True)
            .head(self._window)
            .select(["symbol"])
        )

        symbols = liquidity_screened["symbol"].to_list()
        if not symbols:
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