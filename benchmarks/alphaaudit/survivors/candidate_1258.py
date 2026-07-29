from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks may exhibit stronger performance during specific times of the year. "
        "By identifying and capitalizing on these seasonal effects, we can construct a strategy that "
        "aims to capture higher returns."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            if (
                values[-1] >= max(values)
                and stamp.month == 12
                or stamp.month == 1
                and stamp.day >= 10
            ):
                seasonal_picks.append(symbol)

        if not seasonal_picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(seasonal_picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in seasonal_picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest