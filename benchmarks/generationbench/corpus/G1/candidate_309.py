from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often associated with lower risk and may offer more stable returns."
        " By tilting our portfolio towards these stocks, we aim to reduce overall portfolio volatility."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the 20-day log returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("log_return")
            )
            .sort("session_date", descending=False)
            .group_by("symbol")
            .agg(pl.col("log_return").mean().alias("avg_log_return"))
        )

        # Calculate the volatility as the standard deviation of log returns
        history = (
            history.with_columns(
                pl.col("log_return").std().alias("volatility")
            )
            .sort("volatility", descending=True)
            .head(self._window)
        )

        symbols = [row["symbol"] for row in history.to_dicts()]
        weight = 1.0 / len(symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest