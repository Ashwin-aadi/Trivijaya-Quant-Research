from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves are indicative of strong market sentiment. "
        "A significant price move accompanied by increased volume suggests a robust trend is forming."
    )

    def __init__(self, window: int = 20, min_volume_threshold: float = 1e6) -> None:
        self._window = window
        self._min_volume_threshold = min_volume_threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            symbol_history = history.select(["session_date", "symbol", "close", "volume"])
            symbol_history = symbol_history.with_columns(
                (pl.col("close") - pl.col("close").shift(1)).alias("price_diff"),
                (pl.col("volume") / pl.col("volume").shift(1) - 1.0).alias("vol_ratio"),
            )
            if symbol_history.is_empty():
                continue

            latest_close = float(view.latest_close()[symbol])
            recent_close = float(symbol_history["close"][-2])

            # Check for a significant price move and volume increase
            if (
                abs(latest_close - recent_close) / recent_close > 0.05
                and float(symbol_history["vol_ratio"].filter(pl.col("vol_ratio") > 1).mean()) >= 1.2
            ):
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