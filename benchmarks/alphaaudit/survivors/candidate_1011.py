from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LiquidityScreenedEqualWeighting(Strategy):
    rationale = (
        "Liquidity screening ensures that only the most liquid stocks are considered for "
        "investment. Equal weighting among these stocks promotes diversification and can "
        "help mitigate risks associated with overconcentration in less liquid assets."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        liquidity_screened = (
            history.filter(
                pl.col("volume").rolling_sum(window_size=self._window).over("symbol")
                > 10_000_000
            )
            .group_by("symbol")
            .agg(pl.count().alias("session_count"))
            .sort("session_count", descending=True)
        )

        if liquidity_screened.height < 5:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in liquidity_screened.to_dicts()][:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest