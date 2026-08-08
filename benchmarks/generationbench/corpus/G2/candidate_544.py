from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class PriceLevelReversion(Strategy):
    rationale = (
        "Price reversion to the mean is a fundamental concept in finance. "
        "After an asset experiences an extreme price level, it often tends to revert towards its historical average. "
        "This strategy aims to exploit such reversions by identifying symbols that have deviated significantly from their trailing 20-day average and are ready for reversion."
    )

    def __init__(self, window: int = 20, z_score_threshold: float = 2.0) -> None:
        self._window = window
        self._z_score_threshold = z_score_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = set(view.symbols).intersection(set(closes.columns))
        mean_close = (
            view.history().select(pl.col("symbol"), pl.col("adj_close").mean())
            .with_columns(
                (pl.col("adj_close") - pl.col("adj_close").arr.mean()).alias("deviation"),
                ((pl.col("adj_close") - pl.col("adj_close").arr.mean()) / pl.col("adj_close").std()).alias("z_score"),
            )
            .filter(pl.col("symbol").is_in(symbols))
        )

        if mean_close.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        z_scores = (
            mean_close.group_by("symbol")
            .agg(pl.col("z_score").mean().alias("mean_z_score"))
            .filter(pl.col("mean_z_score").abs() > self._z_score_threshold)
            .select("symbol")
            .to_dict()["symbol"]
        )

        if not z_scores:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(z_scores)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in z_scores}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest