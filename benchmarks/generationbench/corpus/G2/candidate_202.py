from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrend(Strategy):
    rationale = (
        "Certain sectors or industries may exhibit strong seasonal trends due to specific "
        "economic drivers. For instance, tourism-related stocks might experience higher returns "
        "during holiday seasons, while agricultural companies could see increased activity and "
        "prices during planting and harvest times."
    )

    def __init__(self, window: int = 20, lookback_periods: int = 12) -> None:
        self._window = window
        self._lookback_periods = lookback_periods

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_periods)

        if closes.height < self._lookback_periods or len(view.symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        seasonality: dict[str, float] = {}
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._lookback_periods:
                continue
            avg_returns = sum((v / values[i - 1] - 1.0 for i, v in enumerate(values)), start=0)
            seasonality[symbol] = (avg_returns / (self._lookback_periods - 1)) * 100

        sorted_symbols = sorted(seasonality.items(), key=lambda x: abs(x[1]), reverse=True)
        top_symbols = [s for s, _ in sorted_symbols[: self._window]]

        weight = 1.0 / len(top_symbols) if top_symbols else 0
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest