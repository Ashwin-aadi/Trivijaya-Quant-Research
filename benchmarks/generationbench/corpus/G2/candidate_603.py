from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "High-liquidity stocks are typically less susceptible to price manipulation and offer "
        "better trading opportunities. By equally weighting high liquidity stocks, we aim to "
        "capture the benefits of lower transaction costs and higher tradeability."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.select(pl.col("symbol"))
            .with_columns(
                (pl.col("volume").rolling_sum(window_size=self._window) / 100_000).alias("avg_volume")
            )
            .sort("avg_volume", descending=True)
            .head(self._window)
            .select("symbol")
            .to_dict(False)
        )

        if not liquidity_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(liquidity_scores)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [row["symbol"] for row in liquidity_scores]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest