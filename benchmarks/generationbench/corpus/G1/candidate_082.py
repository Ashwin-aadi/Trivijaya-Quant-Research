from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Historical data in the Indian market often exhibits seasonality effects, where "
        "certain months or seasons have historically shown higher returns. This strategy "
        "exploits such patterns by overweighting stocks whose performance is better during "
        "the corresponding month each year."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        month_dict: dict[str, list[float]] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            month = date.fromisoformat(view.history(lookback=self._window)["session_date"][0]).month
            if month not in month_dict:
                month_dict[month] = []
            month_dict[month].append(max(values))

        max_per_month: list[float] = [max(months) for _, months in month_dict.items()]
        picks: list[str] = []

        current_month = stamp.month
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            if max(values) == max_per_month[current_month - 1]:
                picks.append(symbol)

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