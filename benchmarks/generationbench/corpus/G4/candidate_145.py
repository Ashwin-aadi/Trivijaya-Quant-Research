from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion1d(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the phenomenon where stock prices tend to "
        "correct deviations from their historical average levels over short periods. By "
        "identifying stocks with significant price deviations using Z-scores, this strategy "
        "aims to capture profitable reversals."
    )

    def __init__(self, lookback: int = 1, rank_by: str = 'abs_zscore', top_n: int = 50) -> None:
        self._lookback = lookback
        self._rank_by = rank_by
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback:
            return Signal(information_available_at=stamp, weights={})

        mean_close = history.select(
            pl.col("adj_close").mean().alias("mean")
        )
        std_dev_close = history.select(
            (pl.col("adj_close") - pl.col("mean")).stddev().alias("std_dev")
        )

        z_scores = (
            history.join(mean_close, on="session_date", how="inner")
                .join(std_dev_close, on="session_date", how="inner")
                .select(pl.all().drop_nulls())
                .with_columns(
                    (pl.col("adj_close") - pl.col("mean")) / pl.col("std_dev").alias("zscore")
                )
        )

        ranked = z_scores.sort("zscore", descending=True) \
            .group_by("symbol") \
            .agg((pl.col("zscore").abs().rank(method="dense", descending=True)).alias("rank"))

        top_symbols = ranked.select(pl.col("symbol")).head(self._top_n).to_list()[0]
        weights = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest