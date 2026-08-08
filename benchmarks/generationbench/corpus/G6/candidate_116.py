from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedDirectionalMove(Strategy):
    rationale = (
        "This strategy identifies significant price movements accompanied by high trading volumes to capitalize on strong market sentiment and liquidity. It ensures that entries are made during periods of active trading, reducing risk while leveraging the power of volume-confirmed directional moves."
    )

    def __init__(self, window: int = 20, threshold_multiplier: float = 1.5, reversal_threshold: float = 0.6) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier
        self._reversal_threshold = reversal_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            data = history.select(pl.col("symbol"), pl.col("session_date"), pl.col("open"), pl.col("close"), pl.col("volume"))
            open_price = float(data.filter(pl.col("symbol") == symbol).select(pl.col("open")).to_series().values[0])
            close_price = float(data.filter(pl.col("symbol") == symbol).select(pl.col("close")).to_series().values[1])
            volume_today = float(data.filter(pl.col("symbol") == symbol).filter(pl.col("session_date") == stamp).select(pl.col("volume")).to_series().values[0])
            prev_volume_avg = float(data.filter(pl.col("symbol") == symbol).group_by("symbol").agg((pl.col("volume").mean()).alias("avg_volume")).select("avg_volume").to_series().values[0])

            if close_price > open_price and volume_today >= self._threshold_multiplier * prev_volume_avg:
                picks.append(symbol)

        picks = picks[:15]
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