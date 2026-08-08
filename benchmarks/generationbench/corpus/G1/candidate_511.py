from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates that the market is consolidating and may soon breakout. "
        "Identifying stocks with increased range can signal potential upcoming movements."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        ranges: dict[str, float] = {}
        for symbol in view.symbols:
            high_low_diff = (
                history.filter(pl.col("symbol") == symbol)
                .select(
                    (pl.col("high").max() - pl.col("low").min()).alias("range")
                )
                .collect()["range"]
            )[0]
            ranges[symbol] = float(high_low_diff)

        sorted_ranges = sorted(ranges.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_ranges[:5]]

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