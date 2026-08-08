from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqualWeighting(Strategy):
    rationale = (
        "High liquidity stocks are generally more efficient in price discovery and have lower "
        "transaction costs. By investing equally in the most liquid stocks, we can potentially "
        "benefit from reduced slippage and higher trading volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.group_by("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
        )

        sorted_symbols = liquidity_scores.sort("liquidity_score", descending=True)["symbol"].to_list()
        top_n_symbols = sorted_symbols[:5]  # Assuming we always take the top 5 most liquid symbols

        weights = {s: 1.0 / len(top_n_symbols) for s in top_n_symbols}
        return Signal(
            information_available_at=stamp,
            weights={s: float(weights[s]) for s in view.symbols if s in weights},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest