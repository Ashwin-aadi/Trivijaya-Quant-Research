from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class DualMomentumAndVolume(Strategy):
    rationale = (
        "Combining momentum and volume signals can provide a more robust strategy. "
        "High momentum stocks with increasing trading volumes often indicate strong buying interest."
    )

    def __init__(self, window1: int = 20, window2: int = 5) -> None:
        self._window1 = window1
        self._window2 = window2

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=max(self._window1, self._window2))
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        volume = [float(v) for v in history["volume"].drop_nulls().to_list()]

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes or symbol not in volume:
                continue
            adj_closes = closes[history.columns.index(symbol)]
            volumes = volume[history.columns.index(symbol)]

            if len(adj_closes) < self._window1 or len(volumes) < self._window2:
                continue

            momentum_score = (adj_closes[-1] - adj_closes[0]) / max(adj_closes[0], 1e-6)
            volume_score = sum(volumes[-self._window2:]) / max(sum(volumes), 1)

            if momentum_score > 0.05 and volume_score > 0.8:
                picks.append(symbol)

        picks = picks[:5]  # Select top 5 stocks based on the composite score
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