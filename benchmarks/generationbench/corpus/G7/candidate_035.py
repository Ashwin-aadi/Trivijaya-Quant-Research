from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityLiquidity(Strategy):
    rationale = (
        "Combining 5-day rolling standard deviation of closing prices with a 7-day moving average "
        "of trading volume creates a composite signal that captures both volatility and liquidity. "
        "In the Indian market, stocks with high liquidity tend to be more stable, while higher "
        "volatility can indicate greater potential for price movement."
    )

    def __init__(self, window_vol: int = 5, window_volume: int = 7, top_n: int = 5) -> None:
        self._window_vol = window_vol
        self._window_volume = window_volume
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window_vol, self._window_volume))
        if history.is_empty() or history.height < max(self._window_vol, self._window_volume):
            return Signal(information_available_at=stamp, weights={})

        closes = history.select(["symbol", "session_date", pl.col("adj_close").alias("close")])
        volume = history.select(["symbol", "session_date", pl.col("volume").alias("vol")])

        vol_scores = (
            closes.with_columns(
                (pl.col("close").rolling_std(window_size=self._window_vol)).alias(f"std_{self._window_vol}")
            )
            .group_by("symbol")
            .agg(pl.col(f"std_{self._window_vol}").mean().alias("volatility_score"))
        )

        volume_scores = (
            volume.with_columns(
                (pl.col("vol").rolling_mean(window_size=self._window_volume)).alias(f"avg_vol_{self._window_volume}")
            )
            .group_by("symbol")
            .agg(pl.col(f"avg_vol_{self._window_volume}").mean().alias("liquidity_score"))
        )

        combined_scores = vol_scores.join(volume_scores, on="symbol", how="inner")

        if combined_scores.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_stocks = (
            combined_scores.sort(by="volatility_score", descending=True)
            .sort(by="liquidity_score", descending=True)
            .select("symbol")
            .head(self._top_n)
            .to_dict(as_series=False)
        )

        weight = 1.0 / self._top_n
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_stocks["symbol"]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest