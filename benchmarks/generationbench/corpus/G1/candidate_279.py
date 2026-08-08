from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity-weighted equal weighting involves assigning weights based on the "
        "liquidity of each stock. More liquid stocks are given a higher weight to ensure"
        " that the portfolio is not overly exposed to less tradable securities."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
            .agg(
                (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"),
            )
            .sort("liquidity_score", descending=True)
            .select(pl.col("symbol"))
        )

        if liquidity_scores.height < 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(row["symbol"]) for row in liquidity_scores.rows()]
        total_liquid_volume = float(liquidity_scores.select(pl.sum("volume")).item())
        weights = {s: (float(liquidity_scores.filter(pl.col("symbol") == s).select("volume").item()) / total_liquid_volume) for s in symbols}

        return Signal(
            information_available_at=stamp, weights={s: w for s, w in weights.items()}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest