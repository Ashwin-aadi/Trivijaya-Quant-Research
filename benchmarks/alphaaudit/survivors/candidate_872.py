from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong momentum and can be a reliable "
        "signal for entry into positions. This strategy identifies symbols with significant"
        " volume increase on their recent directionally strong day."
    )

    def __init__(self, window: int = 10, min_volume_increase: float = 2.0) -> None:
        self._window = window
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns and volumes
        history = (
            history.with_columns(
                (pl.col("close") / pl.col("adj_close").shift(1) - 1.0).alias("r"),
                (pl.col("volume")).alias("v")
            )
            .sort("session_date", descending=True)
            .with_columns((pl.col("r") * pl.col("v")).alias("rv"))
        )

        # Identify the most recent strong direction
        strong_day = history.sort("rv", descending=True).head(1)

        if strong_day.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol = str(strong_day["symbol"][0])
        weight = 1.0

        return Signal(
            information_available_at=stamp,
            weights={symbol: weight}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest