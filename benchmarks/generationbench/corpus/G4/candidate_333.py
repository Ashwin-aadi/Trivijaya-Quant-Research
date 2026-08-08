from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "This strategy exploits the low-volatility tilt effect by selecting stocks with "
        "lower historical volatility. Lower volatility stocks tend to outperform high-volatility "
        "stocks over time due to reduced risk premiums and lower susceptibility to price declines."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        realized_volatility = (
            history.select(
                pl.col("symbol"),
                (pl.col("close").shift(-1) / pl.col("open") - 1).alias("log_return")
            )
            .group_by("symbol")
            .agg((pl.col("log_return").std().alias("volatility")))
            .sort("volatility", descending=False)
        )

        symbols = realized_volatility["symbol"].to_list()[: self._top_n]
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