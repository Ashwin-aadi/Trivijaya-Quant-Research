from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity is a key indicator of marketability. By selecting the most liquid stocks, "
        "we can ensure that trades are executed efficiently without impacting prices significantly."
    )

    def __init__(self, lookback: int = 20) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
                   .agg((pl.col("volume").mean()).alias("avg_volume"))
                   .sort("avg_volume", descending=True)
        )

        picks: list[str] = [row["symbol"] for row in liquidity_scores.to_dicts() if row["avg_volume"]]
        weight = 1.0 / len(picks) if picks else 0.0
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest