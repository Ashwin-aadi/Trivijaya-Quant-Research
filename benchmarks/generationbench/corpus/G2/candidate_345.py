from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks have historically outperformed high-volatility stocks due to the "
        "reduction in idiosyncratic risk. By tilting our portfolio towards low volatility, we aim "
        "to capture this premium."
    )

    def __init__(self, lookback: int = 60) -> None:
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate log returns for each symbol
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("log_return")
            )
            .group_by("symbol")
            .agg(pl.col("log_return").mean().alias("mean_log_return"))
        )

        # Sort symbols by mean log return (i.e., lower volatility)
        sorted_symbols = history.sort("mean_log_return", descending=False)

        # Select top 5 low-volatility symbols
        picks: list[str] = [row["symbol"] for row in sorted_symbols.to_dicts()][:5]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest