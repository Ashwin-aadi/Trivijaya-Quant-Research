from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "This strategy exploits the empirical observation that lower-volatility stocks tend to outperform higher-volatility counterparts over long periods. "
        "By tilting our portfolio towards less volatile equities, we aim to capture this effect and potentially achieve superior risk-adjusted returns."
    )

    def __init__(self, lookback: int = 60, percentile: float = 0.5) -> None:
        self._lookback = lookback
        self._percentile = percentile

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            history.select(pl.col("adj_close").shift(-1) / pl.col("adj_close") - 1.0)
            .filter(~pl.col("session_date").is_null())
            .select(pl.col("symbol"), (pl.col("close") / pl.col("open").shift(1) - 1.0).alias("return"))
        )
        returns = (
            returns.groupby("symbol")
            .agg((pl.col("return").std().alias("volatility")))
            .sort("volatility", descending=False)
        )

        top_percentile_count = int(len(symbols) * self._percentile)
        selected_symbols = [row[0] for row in returns.head(top_percentile_count).to_numpy()]
        weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}

        return Signal(
            information_available_at=stamp,
            weights=weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest