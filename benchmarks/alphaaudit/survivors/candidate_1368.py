from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for recent winners to continue outperforming "
        "recent losers. This strategy buys top performers and sells bottom performers based on their "
        "returns over the last 20 trading days."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history_with_returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("returns")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Calculate cumulative returns over the last `window` days
        cum_returns = (
            history_with_returns.group_by("symbol")
                                 .agg(
                                     pl.col("returns").sum().alias("cumulative_return"),
                                 )
                                 .with_columns(pl.col("cumulative_return") / self._window)
        )

        top_symbols = cum_returns.sort("cumulative_return", descending=True).select(
            "symbol"
        ).to_series().to_list()[:5]

        bottom_symbols = cum_returns.sort("cumulative_return").select(
            "symbol"
        ).to_series().to_list()[:5]

        weight_top = 0.4 / len(top_symbols)
        weight_bottom = -0.2 / len(bottom_symbols)

        weights: dict[str, float] = {}
        for symbol in top_symbols:
            weights[symbol] = weight_top
        for symbol in bottom_symbols:
            weights[symbol] = weight_bottom

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