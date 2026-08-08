from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in the recent past to continue outperforming those with poor performance. "
        "This strategy ranks symbols based on their returns over a short window and "
        "allocates capital to the top performers."
    )

    def __init__(self, lookback_window: int = 10, num_top_symbols: int = 5) -> None:
        self._lookback_window = lookback_window
        self._num_top_symbols = num_top_symbols

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history_with_returns = (
            history
            .with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .select(["symbol", "session_date", "return"])
        )

        # Calculate cumulative returns over the lookback window
        cum_returns = (
            history_with_returns
            .group_by("symbol")
            .agg(
                pl.col("return").sum().alias("cumulative_return"),
            )
        )

        # Rank symbols by their cumulative return
        ranked_symbols = (
            cum_returns.sort(pl.col("cumulative_return"), descending=True)
            .head(self._num_top_symbols)
            .select(["symbol"])
            .to_dict(as_series=False)
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight_per_symbol = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight_per_symbol for s in ranked_symbols["symbol"]},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest