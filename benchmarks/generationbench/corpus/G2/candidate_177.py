from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is often attributed to risk aversion among investors, who are willing to pay a premium for less volatile assets."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute daily returns for each symbol
        history = (
            history
            .with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .drop_nulls()
        )

        volatilities = (
            history.group_by("symbol")
            .agg(pl.col("return").std().alias("volatility"))
            .sort("volatility", descending=False)
        )

        if volatilities.is_empty():
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [row["symbol"] for row in volatilities.to_dicts()[:10]]
        weight = 1.0 / len(top_symbols)

        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest