from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates increased market uncertainty and can lead to"
        " breakout opportunities. By identifying symbols with reduced price ranges,"
        " we aim to capture potential reversals or trend extensions."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if symbol in history["symbol"].unique()]
        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        compressed: dict[str, float] = {}
        for symbol in symbols:
            range_values = (
                history.filter(pl.col("symbol") == symbol)
                       .select(
                           (pl.col("high") - pl.col("low")).alias("range")
                       )
                       .sort("session_date", descending=False)
            )

            if range_values.height < self._window:
                continue
            latest_mean = float(range_values.select("range").tail(1)["range"].to_list()[0])
            current_range = float(
                history.filter(pl.col("symbol") == symbol).select("high").tail(1)["high"].to_list()[0] -
                history.filter(pl.col("symbol") == symbol).select("low").tail(1)["low"].to_list()[0]
            )
            if latest_mean / current_range < 0.5:
                compressed[symbol] = latest_mean

        if not compressed:
            return Signal(information_available_at=stamp, weights={})

        mean_compression = sum(compressed.values()) / len(compressed)
        picks: list[str] = [symbol for symbol, comp in compressed.items() if comp <= mean_compression]
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest