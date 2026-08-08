from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityWeightedEqual(Strategy):
    rationale = (
        "Liquidity screening can help identify stocks that are more likely to be traded "
        "without significant price impact. Equal weighting among these liquid stocks can "
        "lead to a balanced portfolio."
    )

    def __init__(self, window: int = 20) -> None:
        assert self._window > 0 and self._window < 100, "Window must be between 1 and 99"
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.groupby("symbol")
                   .agg([
                       (pl.col("volume").sum() / pl.col("adj_close").mean()).alias("liquidity_score"),
                       (pl.col("close") - pl.col("open")).abs().sum().alias("price_range_sum"),
                   ])
        ).sort(by="liquidity_score", descending=True)

        if liquidity_screened.height < 1:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in liquidity_screened.to_dicts()[:20]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].cast(pl.Date).max()
    assert isinstance(newest, date), "Session date must be of Date type"
    return newest