from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Seasonality effects can be exploited by identifying periods during the year "
        "when certain stocks tend to outperform. This strategy focuses on trading "
        "during these favorable times based on historical price data."
    )

    def __init__(self, window: int = 10, threshold_high: float = 0.2, threshold_low: float = -0.2) -> None:
        self._window = window
        self._threshold_high = threshold_high
        self._threshold_low = threshold_low

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) < 20:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            mean_return = sum(values[-10:] - values[:-10]) / 10.0
            if mean_return > self._threshold_high or mean_return < self._threshold_low:
                picks.append(symbol)

        picks = picks[:5]
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