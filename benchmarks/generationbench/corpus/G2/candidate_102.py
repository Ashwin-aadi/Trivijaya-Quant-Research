from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Securities with higher relative strength to the broader market tend to continue their "
        "trend. This is based on the assumption that strong stocks will outperform weak ones over time."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(view.symbols) <= 1:
            return Signal(information_available_at=stamp, weights={})

        benchmark_close = float(view.latest_close()[view.as_of.strftime("%Y-%m-%d")])
        strength_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            close_series = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(close_series) < self._window or symbol == "NIFTY":
                continue
            factor = (close_series[-1] - close_series[0]) / benchmark_close
            strength_factors[symbol] = factor

        sorted_strengths = sorted(strength_factors.items(), key=lambda item: item[1], reverse=True)
        top_n_symbols = [symbol for symbol, _ in sorted_strengths[:5]]
        weights = {s: 1.0 / len(top_n_symbols) for s in top_n_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest