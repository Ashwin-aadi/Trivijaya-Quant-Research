from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "This strategy aims to identify and capitalize on volume-confirmed directional moves. "
        "High-volume days often precede significant price movements due to increased investor "
        "participation and information dissemination. By detecting these patterns, we can benefit "
        "from the lag between volume breakout and price confirmation."
    )

    def __init__(self, window: int = 3, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 3)
        if history.height < self._window + 3:
            return Signal(information_available_at=stamp, weights={})

        volume_changes = []
        for symbol in view.symbols:
            if symbol not in history.columns or len(history[symbol].select("volume").to_list()) != self._window + 3:
                continue

            prev_volume = float(history[symbol]["volume"].item(-1))
            curr_volume = float(history[symbol]["volume"].item(-2))
            volume_change_percent = ((curr_volume - prev_volume) / prev_volume) * 100
            if volume_change_percent > 20:
                price_changes = [float(price) for price in history[symbol][["close", "open"]].to_list()[-self._window:]]
                price_change_after_consolidation = (price_changes[-1] - min(price_changes[:-2])) / min(price_changes[:-2]) * 100
                if price_change_after_consolidation > 5:
                    volume_changes.append((symbol, volume_change_percent, price_change_after_consolidation))

        if not volume_changes:
            return Signal(information_available_at=stamp, weights={})

        volume_changes.sort(key=lambda x: (x[1], x[2]), reverse=True)
        picks = [symbol for symbol, _, _ in volume_changes[:self._top_n]]
        weight = 1.0 / len(picks) if picks else 0
        return Signal(information_available_at=stamp, weights={s: weight for s in picks})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest