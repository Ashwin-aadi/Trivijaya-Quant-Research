from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedBreakout(Strategy):
    rationale = (
        "This strategy identifies price breakouts that are confirmed by significant trading volume. "
        "High-volume confirmations suggest strong market commitment and potential for sustained price action."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        h_l_range = (history["high"] - history["low"]).alias("h_l_range")
        volume_ratio = (history["volume"] / pl.col("volume").mean().over(history.window_by(["symbol"], length=self._window))).alias("v_v_avg")

        ranked_symbols = (
            history
            .with_columns([h_l_range, volume_ratio])
            .group_by("symbol", maintain_order=True)
            .agg(
                h_l_sum = pl.col("h_l_range").sum(),
                v_v_avg = pl.col("v_v_avg").mean()
            )
            .select(
                "symbol",
                (pl.col("h_l_sum") / pl.col("v_v_avg")).alias("rank")
            )
            .sort("rank", descending=True)
            .to_series()
            .to_list()[:self._top_n]
        )

        if not ranked_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(ranked_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in ranked_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest