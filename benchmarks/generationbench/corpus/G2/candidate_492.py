from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks which have outperformed in the recent "
        "past are likely to continue outperforming. This can be measured by looking at returns "
        "relative to historical means and applying a ranking mechanism."
    )

    def __init__(self, window: int = 20, lookback: int = 60) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.is_empty() or history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the returns for each symbol over the lookback period
        returns = (
            history.select(
                pl.col("symbol"),
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1).alias("return")
            )
            .group_by("symbol")
            .agg(pl.col("return").mean().alias("avg_return"))
        )

        # Rank symbols based on average return
        ranked = returns.sort("avg_return", descending=True).with_columns(
            pl.col("avg_return").rank(method="dense", descending=True).alias("rank")
        )

        if ranked.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        # Select top N symbols based on rank
        picks = ranked.select("symbol")[:self._lookback]["symbol"].to_list()

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest