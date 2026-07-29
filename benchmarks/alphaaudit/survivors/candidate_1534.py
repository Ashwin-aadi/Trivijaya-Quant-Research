from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance during specific months or seasons of the year. "
        "This strategy aims to capitalize on these seasonal patterns by overweighting those symbols that have historically performed well."
    )

    def __init__(self, window: int = 10, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the mean close price over the lookback period
            avg_close = sum(values[-self._window:]) / self._window
            recent_max = max(values[-self._window:])
            recent_min = min(values[-self._window:])

            if recent_max - recent_min > 0.1 * (recent_max + recent_min) and values[-1] >= avg_close:
                picks.append(symbol)

        picks = picks[: self._top_n]
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