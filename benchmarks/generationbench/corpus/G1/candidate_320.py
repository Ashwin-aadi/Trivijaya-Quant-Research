from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeConfirmedMove(Strategy):
    rationale = (
        "A significant price move accompanied by increased volume is often a strong indication "
        "of continuation of the trend. This strategy aims to identify such moves and capitalize on them."
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
            hist_df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            opens = [float(o) for o in hist_df["open"].to_list()]
            closes = [float(c) for c in hist_df["close"].to_list()]

            if len(opens) < self._window:
                continue

            max_close = max(closes)
            min_open = min(opens)

            # Identify significant price move
            if (max_close - min_open) / min_open >= 0.05 and closes[-1] > max_close or \
               (min_open - max_close) / min_open >= 0.05 and closes[-1] < min_open:
                vol = float(hist_df.select(pl.col("volume").sum()).to_numpy()[0][0])
                if vol > sum([float(v) for v in hist_df.select(pl.col("volume"))[:self._window].to_numpy().flatten()]) / self._window * 2:
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