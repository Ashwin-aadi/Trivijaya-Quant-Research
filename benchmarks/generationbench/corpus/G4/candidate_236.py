from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmation(Strategy):
    rationale = (
        "This strategy captures volume-confirmed directional moves in Indian equity markets by "
        "identifying stocks with significant increases in trading volumes over a defined period. "
        "High volume surges are indicative of strong buying or selling pressure from institutional traders."
    )

    def __init__(self, window: int = 5, max_positions: int = 20) -> None:
        self._window = window
        self._max_positions = max_positions

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window * 2)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        avg_volume = (
            history.select(
                pl.col("symbol"),
                (pl.col("volume") / self._window).alias("avg_volume")
            )
            .group_by("symbol")
            .agg(pl.col("avg_volume").mean().alias("m"))
            .with_column(
                (pl.col("volume") - pl.col("avg_volume")) / pl.col("avg_volume").rank(method="dense", descending=True)
                .alias("vol_change_ranked")
            )
        )

        if avg_volume.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = avg_volume.sort("vol_change_ranked", descending=True).select(
            pl.col("symbol").head(self._max_positions)
        ).to_dict(as_series=False)

        weight = 1.0 / len(top_symbols["symbol"])
        return Signal(information_available_at=stamp, weights={s: weight for s in top_symbols["symbol"]})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest