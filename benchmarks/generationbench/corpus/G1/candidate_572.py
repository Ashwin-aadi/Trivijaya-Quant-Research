from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "This strategy identifies symbols that have made a significant upward move with high volume."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            data = (
                history.filter(pl.col("symbol") == symbol)
                          .sort("session_date")
                          .with_columns(
                              (pl.col("close") - pl.col("open")).alias("move"),
                              (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("vol_change")
                          )
            )

            latest_close = data.select(pl.col("adj_close").last().alias("latest"))
            if not latest_close.height:
                continue

            recent_moves = [float(v) for v in data["move"].to_list()]
            recent_vols = [float(v) for v in data["vol_change"].to_list()]

            max_move_index = recent_moves.index(max(recent_moves))
            vol_increase = recent_vols[max_move_index] > 0.1

            if len(recent_moves) >= self._window and max_move_index == (len(recent_moves) - 1) and vol_increase:
                picks.append(symbol)

        picks = picks[:5]
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