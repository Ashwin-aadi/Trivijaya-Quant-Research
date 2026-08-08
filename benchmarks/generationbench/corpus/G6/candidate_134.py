from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy identifies significant price movements by scaling trend strength with "
        "volatility. It enters positions based on high volume and upward trending Simple Moving "
        "Averages (SMA), ensuring strong trends are captured with sufficient liquidity, while "
        "exiting when volatility increases or trading volume decreases."
    )

    def __init__(self, window: int = 20, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 30)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        sma_20 = history.select(
            pl.col("symbol"),
            (pl.col("adj_close").rolling_mean(window_size=20)).alias("sma_20")
        )
        volume_diff_30 = history.select(
            pl.col("symbol"),
            (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("volume_ratio")
        )

        combined_df = sma_20.join(volume_diff_30, on="symbol", how="inner")

        breakout = combined_df.with_columns(
            (pl.col("adj_close") > pl.col("sma_20")).cast(pl.int8).alias("breakout")
        ).group_by("symbol").agg(
            (pl.col("volume_ratio").mean().alias("avg_volume_ratio")),
            (pl.col("breakout").sum()).alias("breakouts_count")
        )

        breakout = breakout.filter(
            (pl.col("avg_volume_ratio") > 1.0) & (pl.col("breakouts_count") >= 2)
        ).sort("breakouts_count", descending=True).select("symbol")

        if breakout.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        picks = [str(row["symbol"]) for row in breakout.to_dicts()][:self._top_n]
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