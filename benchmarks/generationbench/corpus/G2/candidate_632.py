from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirm(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate that a significant number of market participants "
        "are actively engaging in the direction of the move. This can lead to sustained trends and potentially "
        "profitable opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volume_condition = (
            (history["volume"] > history.select(pl.col("volume").quantile(0.75))[-1].item())
            & (history["close"].shift(-1) - history["close"] >= 0)
        ).sum() > self._window / 2

        if not volume_condition:
            return Signal(information_available_at=stamp, weights={})

        strong_moves = (
            history.select(
                pl.col("symbol"),
                (pl.col("close") - pl.col("open")).abs().rank(method="dense", descending=True)
            )
            .sort((pl.col("close") - pl.col("open")).abs(), descending=True)
            .head(10)["symbol"]
        ).to_list()

        if not strong_moves:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(strong_moves)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in strong_moves}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest