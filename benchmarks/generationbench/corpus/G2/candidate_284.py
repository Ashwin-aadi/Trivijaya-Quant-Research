from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Large volume moves can indicate significant changes in market sentiment or the entry of "
        "large traders. A directional move accompanied by high volume is more likely to be a true signal "
        "of underlying strength or weakness than a small move."
    )

    def __init__(self, window: int = 20, threshold_volume_ratio: float = 1.5) -> None:
        self._window = window
        self._threshold_volume_ratio = threshold_volume_ratio

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the percentage change in close
        history = history.with_columns(
            (pl.col("close") / pl.col("close").shift(self._window) - 1.0).alias("price_change")
        )

        # Filter out symbols without enough data
        history = history.drop_nulls().select(["symbol", "session_date", "price_change", "volume"])

        # Identify top volume moves
        high_volume_moves = (
            history.group_by("symbol")
            .agg(
                (pl.col("price_change").mean()).alias("avg_price_change"),
                (pl.col("volume") / pl.col("volume").shift(self._window)).mean().alias("volatility_ratio"),
            )
            .filter(pl.col("volatility_ratio") > self._threshold_volume_ratio)
        )

        if high_volume_moves.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Pick top symbols based on average price change
        picks = (
            high_volume_moves.sort("avg_price_change", descending=True)
            .select(["symbol"])
            .head(self._window)
            .to_series()
            .to_list()
        )

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest