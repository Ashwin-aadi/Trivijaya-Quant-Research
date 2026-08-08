from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Certain stocks in the NIFTY 100 index may exhibit seasonal price movements due to "
        "recurring economic or market factors. For instance, demand for certain sectors might "
        "increase during specific times of the year, leading to higher prices. By identifying "
        "these patterns, we can time our trades to benefit from these predictable moves."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        seasonal_strengths = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history.select(
                pl.col("session_date"), pl.col(symbol)
            ).sort("session_date")
            symbol_closes = [float(v) for v in symbol_history[symbol].to_list()]
            seasonal_strengths[symbol] = (
                max(symbol_closes[-12:]) - min(symbol_closes[-12:])
            ) / (max(symbol_closes[:48]) - min(symbol_closes[:48]))

        top_symbols = sorted(seasonal_strengths.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest