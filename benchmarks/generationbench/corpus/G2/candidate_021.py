from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "This is often attributed to the risk premium investors require for taking on additional risk."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_data = (
            history.select(pl.col("symbol"), pl.col("adj_close").std().alias("volatility"))
            .group_by("symbol")
            .agg(pl.col("volatility").mean().alias("avg_volatility"))
            .sort("avg_volatility", descending=False)
            .to_pandas()
        )

        symbols = volatility_data["symbol"].tolist()[:5]
        weights = {s: 1.0 / len(symbols) for s in symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest