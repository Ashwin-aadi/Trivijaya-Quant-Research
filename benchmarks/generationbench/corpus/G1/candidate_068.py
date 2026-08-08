from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves suggest strong market sentiment. "
        "By focusing on both price and volume, we can identify robust trends that may provide "
        "profitable entry points."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort(by="session_date")
            prices = [float(v) for v in df["adj_close"].drop_nulls().to_list()]
            volumes = [int(v) for v in df["volume"].drop_nulls().to_list()]

            if len(prices) < self._window or len(volumes) < self._window:
                continue

            last_price = prices[-1]
            highest_high = max(prices)
            lowest_low = min(prices)

            if (last_price > highest_high and volumes[-1] > sum(volumes[-self._window :])) or \
                    (last_price < lowest_low and volumes[-1] > sum(volumes[-self._window :])):
                picks.append(symbol)

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