from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue outperforming. This strategy identifies top-performing "
        "stocks based on their returns over a certain lookback period and allocates capital "
        "towards them."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or len(history["symbol"].unique()) < 100:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            history.lazy()
            .group_by("symbol")
            .agg(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("return"),
            )
            .collect()
            .select("symbol", "return")
            .sort("return", descending=True)
            .head(10)["symbol"]
            .to_list()
        )

        if not returns:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(returns)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in returns},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest