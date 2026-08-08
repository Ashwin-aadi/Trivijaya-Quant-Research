from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have outperformed "
        "their peers in recent history to continue outperforming. This phenomenon can be "
        "attributed to various factors such as herding behavior and the persistence of stock returns."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        latest_close = view.latest_close()

        # Compute returns
        returns: pl.DataFrame = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return"),
            )
            .collect()
        )

        # Filter out symbols with insufficient data
        returns = returns.filter(pl.col("return").is_not_null())

        # Rank symbols by return
        ranked_returns = (
            returns
            .sort("return", descending=True)
            .group_by("symbol")
            .agg(
                pl.col("return").rank(method="ordinal", descending=True).alias("rank"),
            )
        )

        top_n_symbols = ranked_returns["symbol"].head(self._look_n)

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        signal_weights = {s: weight for s in top_n_symbols}

        return Signal(
            information_available_at=stamp,
            weights=signal_weights,
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest