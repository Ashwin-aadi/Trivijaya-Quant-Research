from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This is because low-volatility stocks are often considered safer and may offer better risk-adjusted returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate historical returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("mean_return"))
            .with_columns(
                (pl.col("return") / pl.col("return").std() * -1).alias("volatility_adjusted_return")
            )
        )

        # Rank by volatility-adjusted returns
        ranked = history.sort("volatility_adjusted_return", descending=True)
        symbols = [symbol for symbol in view.symbols if symbol in ranked.columns]

        top_symbols = symbols[:5]  # Select top 5 symbols

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