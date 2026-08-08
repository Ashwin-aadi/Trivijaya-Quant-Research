from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity-screened equal weighting involves investing equally in the most liquid "
        "names. Higher liquidity suggests better marketability and potentially lower transaction costs."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_scores = (
            history.groupby("symbol")
                   .agg((pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"))
        )

        symbols = [symbol for symbol in view.symbols if symbol in liquidity_scores["symbol"]]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        liquidity_ranked = (
            liquidity_scores.filter(pl.col("symbol").is_in(symbols))
                             .sort("liquidity_score", descending=True)
                             .select(["symbol"])
        )

        top_symbols = [row["symbol"] for row in liquidity_ranked.rows()]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest