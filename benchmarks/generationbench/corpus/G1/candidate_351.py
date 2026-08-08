from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighing(Strategy):
    rationale = (
        "Liquidity screening ensures that only stocks with sufficient trading volume are "
        "considered for the portfolio. Equal weighting across these stocks can provide a "
        "simple yet effective way to achieve market exposure without bias towards any single stock."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened_history = (
            history
            .group_by("symbol")
            .agg((pl.col("volume").mean().alias("avg_volume")))
            .sort("avg_volume", descending=True)
            .head(10)  # Select top 10 based on average volume
        )

        if liquidity_screened_history.height < 10:
            return Signal(information_available_at=stamp, weights={})

        selected_symbols = [row["symbol"] for row in liquidity_screened_history.iter_rows()]
        equal_weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: equal_weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest