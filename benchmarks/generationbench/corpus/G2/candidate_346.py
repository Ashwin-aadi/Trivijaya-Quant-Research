from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Seasonality in equity markets refers to the tendency of certain stocks or indices "
        "to perform better during specific times of the year. This strategy aims to exploit "
        "such trends by identifying and weighting towards stocks that have historically outperformed "
        "during the same period each year."
    )

    def __init__(self, window: int = 365) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonal_trends: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue

            # Calculate the seasonal trend by comparing current close with historical closes of this date
            current_close_index = _date_to_index(stamp, closes)
            if current_close_index >= 0 and len(values) > current_close_index + 1:
                recent_closes = values[current_close_index - self._window : current_close_index]
                seasonal_trends[symbol] = (values[-1] / max(recent_closes) - 1.0)

        # Filter out symbols with no clear trend
        picks = [symbol for symbol, trend in seasonal_trends.items() if trend > 0.05]

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


def _date_to_index(date: date, df: pl.DataFrame) -> int:
    idx = df.height - 1
    while idx >= 0 and df.row(idx)["session_date"] != date:
        idx -= 1
    return idx if df.row(idx)["session_date"] == date else -1