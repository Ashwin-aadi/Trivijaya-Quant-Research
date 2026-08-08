from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolumeBreakout(Strategy):
    rationale = (
        "Volume-confirmed directional moves can signal a strong change in market sentiment. "
        "When a stock breaks out of its recent range with significant volume, it often "
        "indicates a sustained move, which can provide profit opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            adj_closes = df.select(pl.col("adj_close").to_list())
            volumes = df.select(pl.col("volume").to_list())

            if len(adj_closes) < self._window or len(volumes) < self._window:
                continue

            last_close = float(adj_closes[-1])
            max_close = max([float(c) for c in adj_closes])
            min_close = min([float(c) for c in adj_closes])

            # Check if the close is at the top of its range
            if last_close == max_close:
                breakout_symbols.append(symbol)
            elif last_close == min_close:
                continue

        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select(pl.col("session_date").max()).to_series().to_list()[0]
    assert isinstance(newest, date)
    return newest