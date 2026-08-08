from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in recent periods to continue outperforming. This is based on "
        "the idea that winners stay winners and losers stay losers over short time horizons."
    )

    def __init__(self, lookback_window: int = 30) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Compute the returns over the lookback window
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._lookback_window) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .with_columns((pl.col("return").rank(method="ordinal", descending=True)).alias("rank"))
        )

        # Get the latest close prices for each symbol
        latest_closes = view.latest_close()
        if len(latest_closes) < 1:
            return Signal(information_available_at=stamp, weights={})

        # Filter to get top performers and assign equal weight to them
        top_symbols = [symbol for symbol in history.columns if symbol not in ["session_date", "return", "rank"]]
        ranks = history[top_symbols].select(pl.col("rank").to_list())
        top_n = min(self._lookback_window, len(ranks))
        top_ranks = sorted([float(r) for r in ranks.to_list()[0]], reverse=True)[:top_n]

        # Identify symbols that have the highest rank (best performance)
        top_symbols = [symbol for symbol, rank in zip(top_symbols, ranks.to_list()[0]) if float(rank) in top_ranks]
        weights = {symbol: 1.0 / len(top_symbols) for symbol in top_symbols}

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest