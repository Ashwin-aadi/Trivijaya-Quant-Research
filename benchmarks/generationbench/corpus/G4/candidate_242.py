from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "This strategy aims to capitalize on the relationship between market trends and volatility. "
        "During periods of low volatility, larger positions are taken to capture potential trending "
        "opportunities; during high volatility, trade sizes are reduced to minimize risk."
    )

    def __init__(self, window: int = 20, risk_level_factor: float = 1.5) -> None:
        self._window = window
        self._risk_level_factor = risk_level_factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        # Calculate 20-day standard deviation of daily returns
        returns = (history["adj_close"].to_list()[1:] / history["adj_close"].shift(1).to_list()[:-1]) - 1.0
        volatility_index = pl.DataFrame({"returns": returns}).with_columns(
            (pl.col("returns").std()).alias("volatility")
        ).select(pl.col("volatility")).item()

        # Determine risk level based on volatility index
        risk_level = self._risk_level_factor / volatility_index

        # Identify uptrends and downtrends using moving averages
        ma_high = history["adj_close"].rolling_max(window=self._window).alias("ma_high")
        ma_low = history["adj_close"].rolling_min(window=self._window).alias("ma_low")

        trends = (
            history
            .with_columns(ma_high, ma_low)
            .select(pl.col("session_date"), pl.col("symbol"), (pl.col("adj_close") > pl.lit(0.98 * pl.col("ma_high"))).alias("uptrend"))
            .group_by("symbol")
            .agg(
                (pl.col("ma_high").mean()).alias("avg_ma_high"),
                (pl.col("ma_low").mean()).alias("avg_ma_low"),
                (pl.count().filter(pl.col("uptrend") == True)).alias("num_ascending_days")
            )
        )

        # Rank potential trades based on the calculated risk level and trend
        trends = (
            trends.with_columns(
                (pl.col("num_ascending_days") / self._window).rank(method="dense", descending=True).alias("trend_rank"),
                (1.0 + pl.col("avg_ma_high").std() * -risk_level).alias("position_size")
            )
        )

        # Select top 5 symbols for trade
        picks = trends.sort("position_size", descending=False).select("symbol").head(5).to_list()

        if not picks:
            return Signal(information_available_at=stamp, weights={})

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