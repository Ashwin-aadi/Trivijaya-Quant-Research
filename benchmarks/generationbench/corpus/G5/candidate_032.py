from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate a strong market sentiment. "
        "A significant volume increase during a directional move suggests "
        "a high probability of continued trend."
    )

    def __init__(self, window: int = 10, min_volume_increase: float = 0.5) -> None:
        self._window = window
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            close_ratio_df = (history
                              .filter(pl.col("symbol") == symbol)
                              .select(
                                  pl.col("session_date"),
                                  (pl.col("close") / pl.col("adj_close").shift(1) - 1).alias("r")
                              )
                              .sort("session_date")
            )

            volume_ratio_df = (history
                               .filter(pl.col("symbol") == symbol)
                               .select(
                                   pl.col("session_date"),
                                   (pl.col("volume") / pl.col("volume").shift(1)).alias("v_ratio")
                               )
                               .sort("session_date")
            )

            if close_ratio_df.height < self._window or volume_ratio_df.height < self._window:
                continue

            directional_move = close_ratio_df.select(pl.col("r")).all().item()
            volume_confirmation = volume_ratio_df.select(pl.col("v_ratio")).all().item()

            if (
                abs(directional_move) > 0.02
                and abs(volume_confirmation - 1.0) > self._min_volume_increase
            ):
                picks.append(symbol)

        picks = list(set(picks))[:5]
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