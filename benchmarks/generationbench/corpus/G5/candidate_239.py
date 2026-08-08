from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeSignalStrategy(Strategy):
    rationale = (
        "This strategy aims to leverage a combination of recent price momentum and volume "
        "activity. High-volume days with high closing prices are indicative of strong buying "
        "pressure, suggesting potential upward movement."
    )

    def __init__(self, window: int = 20, threshold: float = 0.95) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.symbol.to_list():
                continue
            open_price = [float(o) for o in history.filter(pl.col("symbol") == symbol)["open"].to_list()]
            close_price = [float(c) for c in history.filter(pl.col("symbol") == symbol)["close"].to_list()]
            volume = [float(v) for v in history.filter(pl.col("symbol") == symbol)["volume"].to_list()]

            if len(open_price) < self._window or len(close_price) < self._window or len(volume) < self._window:
                continue

            recent_high_volume_day = volume.index(max(volume))
            recent_high_close = max(close_price)
            open_price_on_high_volume_day = open_price[recent_high_volume_day]

            if open_price_on_high_volume_day > 0.95 * recent_high_close and volume[-1] == max(volume):
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