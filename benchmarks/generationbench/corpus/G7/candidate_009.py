from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility tilting has been shown to enhance returns in equity markets by "
        "systematically overweighting low-volatility stocks. This strategy calculates the 20-day average "
        "volatility of each stock and selects the lowest-volatility names for investment."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
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
            volatility = _calculate_volatility(values)
            picks.append((symbol, volatility))

        sorted_picks = sorted(picks, key=lambda x: x[1])
        top_n_symbols = [p[0] for p in sorted_picks[: self._top_n]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(prices: list[float]) -> float:
    mean_price = sum(prices) / len(prices)
    squared_diffs = [(p - mean_price) ** 2 for p in prices]
    variance = sum(squared_diffs) / (len(prices) - 1)
    return variance ** 0.5