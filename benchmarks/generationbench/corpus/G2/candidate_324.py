from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "relative to their peers in the past period to continue outperforming. This strategy "
        "identifies top performers and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.closes().with_column(pl.col("session_date").alias("symbol"))
        latest_closes = (
            latest_closes.select(["symbol", "adj_close"])
            .sort("symbol")
            .collect()
            .to_dict(False)
        )

        # Calculate returns for each symbol
        history = history.with_columns(
            (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return")
        )
        mean_returns = (
            history.select(pl.all().exclude("symbol", "session_date"))
            .mean()
            .to_dict(False)
        )

        # Rank symbols by return
        ranked_symbols = (
            history.group_by("symbol")
            .agg(
                (pl.col("return").rank(method="max", descending=True).alias("rank"))
            )
            .sort("rank")
            .select(["symbol"])
            .collect()
            .to_dict(False)
        )

        top_symbols = [s[0] for s in ranked_symbols[:10]]

        # Create weights
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest