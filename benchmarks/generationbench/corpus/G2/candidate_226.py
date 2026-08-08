from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalBreakout(Strategy):
    rationale = (
        "Seasonality can arise from predictable changes in economic or market conditions. For "
        "example, certain stocks may show higher returns during specific times of the year due to "
        "seasonal factors like holiday spending or agricultural cycles. A breakout strategy applied"
        " around these seasonal peaks could generate abnormal returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_dict: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the breakout condition based on historical close prices
            last_close = values[-1]
            max_close = max(values)
            if last_close == max_close and (values.index(max_close) + 1) % 20 == 0:  # Assuming a 20-day window
                seasonality_dict[symbol] = max_close

        picks = sorted(seasonality_dict.items(), key=lambda x: x[1], reverse=True)[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={p[0]: weight for p in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest