from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "A significant increase in volume on a directional move can indicate strong momentum "
        "and potential continuation of the trend. This strategy identifies such moves to "
        "enter trades with high confidence."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate the daily price change and volume
        price_changes = (
            history.with_columns(
                (pl.col("adj_close").shift(-1) - pl.col("adj_close")) / pl.col("adj_close").shift(-1).fill_null(1.0)
                .alias("price_change"),
            )
            .sort("session_date", descending=False)
            .with_columns((pl.col("volume") - pl.col("volume").shift(1)).alias("volume_change"))
        )

        # Identify the most recent price change
        latest_price_change = price_changes.select(
            pl.col("symbol"), "price_change"
        ).filter(pl.col("session_date") == stamp).to_dict(as_series=False)

        if not latest_price_change:
            return Signal(information_available_at=stamp, weights={})

        # Find symbols with significant volume change
        volume_change_threshold = 100000  # Arbitrary threshold for volume change
        significant_moves = (
            price_changes.filter(
                (pl.col("volume_change") > volume_change_threshold)
                & (pl.col("price_change").abs() > 0.05)
            )
            .group_by("symbol")
            .agg(pl.col("session_date").max().alias("most_recent_date"))
            .sort("most_recent_date", descending=True)
            .head(self._window)
        )

        # Filter out symbols that do not meet the criteria
        picks: list[str] = [
            symbol for (symbol, _) in significant_moves.to_dicts()
        ]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest