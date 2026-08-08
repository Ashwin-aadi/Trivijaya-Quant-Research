from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "Historical empirical evidence suggests that stocks with lower volatility tend to "
        "outperform those with higher volatility over time. This strategy exploits this "
        "phenomenon by selecting and weighting stocks based on their 12-month trailing "
        "volatility, thereby capturing the 'low-volatility premium'."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(sym) for sym in view.symbols]
        daily_returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("ret")
            )
            .group_by("symbol")
            .agg(pl.col("ret").mean().alias("avg_ret"), pl.col("ret").std().alias("vol"))
            .with_columns(
                (pl.col("vol") / pl.col("vol").max() * 100).alias("rank")
            )
        )

        if daily_returns.height < len(symbols):
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = daily_returns.sort("rank", descending=False)["symbol"].to_list()
        top_n = min(len(sorted_symbols), 200)
        weight = 1.0 / top_n
        return Signal(
            information_available_at=stamp,
            weights=dict(zip(sorted_symbols[:top_n], [weight] * top_n)),
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest