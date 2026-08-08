from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion10d(Strategy):
    rationale = (
        "Short-horizon mean reversion assumes that prices which are far from their recent "
        "average will revert to it. This strategy identifies symbols whose price has moved "
        "significantly away from the 10-day average and buys or sells accordingly."
    )

    def __init__(self, window: int = 10) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 1)

        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = [float(v) for v in view.closes().drop_nulls().to_list()]
        mean_close = sum(recent_closes[-self._window:]) / self._window
        z_scores = [
            (recent_closes[i] - mean_close) / (0.6745 * abs(recent_closes[i] - recent_closes[i-1])) if recent_closes[i-1] != 0 else 0
            for i in range(self._window, len(recent_closes))
        ]

        picks: list[str] = [symbol for symbol, z_score in zip(view.symbols, z_scores) if abs(z_score) > 2.5]
        
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