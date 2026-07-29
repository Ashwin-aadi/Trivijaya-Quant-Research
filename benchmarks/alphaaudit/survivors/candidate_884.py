from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate strong market sentiment and can "
        "potentially lead to continuation of the trend. By identifying such moves, we aim "
        "to capitalize on the momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol = _find_volume_confirmed_move(history)

        if not symbol:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _find_volume_confirmed_move(history: pl.DataFrame) -> str | None:
    symbols = history["symbol"].unique().to_list()

    for symbol in symbols:
        hist = history.filter(pl.col("symbol") == symbol).sort("session_date")
        if hist.is_empty():
            continue

        closes = [float(v) for v in hist["close"].to_list()]
        volumes = [int(v) for v in hist["volume"].to_list()]

        up_days = 0
        down_days = 0
        max_up_day_volume = 0
        min_down_day_volume = float("inf")

        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                up_days += 1
                max_up_day_volume = max(max_up_day_volume, volumes[i])
            elif closes[i] < closes[i - 1]:
                down_days += 1
                min_down_day_volume = min(min_down_day_volume, volumes[i])

        if up_days > 0 and down_days == 0:
            return symbol

        if down_days > 0 and up_days == 0:
            return symbol

    return None