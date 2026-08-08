from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "By tilting the portfolio towards lower volatility names, we aim to capture this premium."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)

        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Compute log returns for each symbol
        log_returns = (closes.melt().with_columns(
            (
                pl.col("value") / pl.col("value").shift(1) - 1.0
            ).alias("log_return")
        )).filter(pl.col("variable").is_not_null())

        # Calculate rolling standard deviation over the last `window` days
        volatilities = log_returns.group_by(
            "variable", maintain_order=True
        ).agg(
            pl.col("log_return").std().alias("volatility")
        )

        # Get symbols with lowest volatilities
        sorted_symbols = (
            volatilities.sort("volatility", descending=False)
            .select(["variable"])
            .to_series()
            .to_list()[:5]
        )

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Allocate equally among the low-volatility symbols
        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest