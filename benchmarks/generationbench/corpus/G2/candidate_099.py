from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighing(Strategy):
    rationale = (
        "Higher liquidity can indicate better marketability and reduced trading costs. "
        "Equally weighting securities based on their liquidity can potentially lead to a "
        "strategy that benefits from the combined flow of these liquid assets."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume").rolling_sum(window_size=self._window) / 100_000).alias("avg_volume"),
                (pl.col("adj_close") * pl.col("volume")).sum().alias("total_value")
            )
            .with_columns(
                ((pl.col("total_value") / pl.col("avg_volume")).rank(method="dense", descending=True) + 1).alias("liquidity_score")
            )
            .sort("liquidity_score")
        )

        if liquidity_scores.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = liquidity_scores.select(pl.col("symbol"))[:self._window]
        weight = 1.0 / self._window
        weights = {str(symbol): weight for symbol in top_symbols["symbol"].to_list()}

        return Signal(
            information_available_at=stamp,
            weights=weights
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest