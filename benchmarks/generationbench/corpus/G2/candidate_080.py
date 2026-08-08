from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "Volume-confirmed directional moves indicate a strong consensus in the market. "
        "Such moves are often followed by continued price appreciation due to higher demand. "
        "By identifying and investing in these moves, we aim to capture this momentum."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            df = history[[symbol, "session_date"]]
            if df.height < self._window:
                continue

            # Calculate the directional move and volume change
            close_changes = [float(v) for v in df["adj_close"].to_list()[1:]]
            vol_changes = [int(v) for v in df["volume"].to_list()[1:]]

            # Check if there's a significant directional move followed by increased volume
            for i in range(len(close_changes)):
                if close_changes[i] > 0 and vol_changes[i + 1] > vol_changes[i]:
                    picks.append(symbol)
                    break

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