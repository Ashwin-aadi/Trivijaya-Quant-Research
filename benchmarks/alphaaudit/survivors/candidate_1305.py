from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to have higher average returns over time. By tilting the "
        "portfolio towards low-volatility stocks, we aim to capture this risk premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        mean_volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close").std() / pl.col("adj_close").mean()).alias("volatility")
            )
            .group_by("symbol")
            .agg(pl.col("volatility").mean().alias("avg_volatility"))
            .sort("avg_volatility", descending=False)
            .to_pandas()
        )

        symbols = view.symbols
        top_n_symbols = mean_volatility["symbol"].head(5).tolist()

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest