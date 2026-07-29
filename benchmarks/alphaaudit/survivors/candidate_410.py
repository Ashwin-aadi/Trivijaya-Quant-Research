from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy follows trends by leveraging volatility. High volatility indicates "
        "increased market uncertainty or momentum, suggesting we should maintain a position in "
        "the most volatile symbols to capture potential upside."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility = (
            history.group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).mean().alias("return_ratio"),
                ((pl.col("high") - pl.col("low")) / pl.col("close")).std().alias("volatility"),
            )
            .sort("volatility", descending=True)
        )

        top_symbols = symbol_volatility.select(["symbol", "volatility"])[: self._window]["symbol"].to_list()
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(top_symbols, [weight] * len(top_symbols))),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest