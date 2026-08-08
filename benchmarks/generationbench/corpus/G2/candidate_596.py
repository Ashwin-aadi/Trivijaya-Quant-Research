from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume breaks in price direction are often indicative of significant buying or "
        "selling pressure. These moves can lead to sustained trends and potentially higher returns."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        history = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("return")
            )
            .sort("session_date", descending=False)
            .drop_nulls()
        )

        # Identify the largest volume day
        max_volume_day = (
            history.select(
                pl.col("volume").max().alias("max_volume"),
                (pl.col("session_date") == pl.col("session_date").shift(-1).filter(pl.col("volume") == pl.col("max_volume"))).alias("is_max_volume")
            )
            .with_columns(pl.when(pl.col("is_max_volume")).then(pl.col("session_date")).otherwise(None).alias("max_volume_date"))
            .select(
                pl.col("max_volume_date").first().alias("max_volume_date"),
                pl.col("return").filter(pl.col("session_date") == pl.col("max_volume_date")).first().alias("max_return")
            )
        )

        if max_volume_day.is_empty() or max_volume_day.height < 1:
            return Signal(information_available_at=stamp, weights={})

        # Determine the breakout direction
        breakout_direction = "bullish" if max_volume_day["max_return"] > 0 else "bearish"

        # Find symbols that confirmed the move with high volume on subsequent days
        confirmation_threshold = history.select(pl.col("volume").quantile(0.95)).first().item()
        confirmation_symbols = (
            history.filter(
                (pl.col("session_date") > max_volume_day["max_volume_date"])
                & (pl.col("volume") >= confirmation_threshold)
            )
            .group_by(["symbol"])
            .agg(pl.col("return").mean().alias("avg_return"))
            .filter(pl.col("avg_return").abs() > 0.02)  # 2% minimum move as a threshold
        )

        if confirmation_symbols.is_empty():
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / confirmation_symbols.height
        return Signal(
            information_available_at=stamp,
            weights={
                symbol: weight for symbol in confirmation_symbols["symbol"].to_list()
            },
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest