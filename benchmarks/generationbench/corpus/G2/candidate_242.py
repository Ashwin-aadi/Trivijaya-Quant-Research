from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy exploits the phenomenon that high-volatility stocks are more likely to "
        "exhibit persistent trends. By scaling our position sizes with volatility, we can "
        "capitalize on these trends while managing risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
            .select(["symbol", "session_date", "r"])
        )

        # Calculate rolling volatility
        vol = (
            returns.groupby("symbol")
            .agg(
                (pl.col("r").rolling_std(self._window, closed="both")).alias("volatility")
            )
            .sort("session_date", descending=False)
            .select(["symbol", "volatility"])
        )

        # Rank symbols by volatility
        ranked = vol.with_columns(
            (pl.col("volatility").rank(method="dense", descending=True)).alias("rank")
        ).select(["symbol", "rank"])

        top_symbols = (
            ranked.sort("rank", descending=False)
            .head(5)
            .get_column("symbol")
            .to_list()
        )

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest