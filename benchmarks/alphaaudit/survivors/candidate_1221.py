from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting is a strategy that seeks to outperform by overweighting "
        "lower-volatility stocks. Historically, lower-volatility stocks tend to have higher"
        " returns over long horizons."
    )

    def __init__(self, lookback_period: int = 60) -> None:
        self._lookback_period = lookback_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_period)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the 20-day rolling standard deviation for each symbol
        volatilities = (
            history.lazy()
            .group_by("symbol")
            .agg(
                pl.col("adj_close").rolling_std(window_size=20).alias("volatility"),
            )
            .select(["symbol", "volatility"])
            .collect()
        )

        # Sort symbols by their rolling standard deviation in ascending order
        sorted_symbols = volatilities.sort("volatility").to_pandas()["symbol"].tolist()

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n = min(len(sorted_symbols), 5)
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols[:top_n]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest