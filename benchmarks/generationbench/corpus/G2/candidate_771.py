from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Highly liquid stocks are more likely to be driven by market forces rather than "
        "influences from specific investors. By equal-weighting these stocks, we aim to "
        "benefit from the liquidity and avoid the biases that come with concentrated positions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history
            .group_by("symbol")
            .agg(pl.col("volume").mean().alias("avg_volume"))
            .sort("avg_volume", descending=True)
            .head(self._window)["symbol"]
            .to_list()
        )

        if len(liquidity_scores) < self._window:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_scores)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in liquidity_scores},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest