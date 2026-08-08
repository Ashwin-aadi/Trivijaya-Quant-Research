from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Relative strength (RS) identifies stocks that are outperforming the market on a trend basis. "
        "Strong relative performance suggests positive fundamentals or momentum, which can generate returns."
    )

    def __init__(self, window: int = 60, top_n: int = 10) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        mean_close = closes.mean().select(
            [pl.col(col).mean() for col in closes.columns[1:]]
        )
        symbol_mean = {symbol: float(mean_close[symbol]) for symbol in view.symbols}

        relative_strengths = []
        for symbol in view.symbols:
            latest_close = float(view.latest_close()[symbol])
            mean_close_value = symbol_mean[symbol]
            if mean_close_value == 0:
                continue
            strength = (latest_close / mean_close_value) - 1.0
            relative_strengths.append((symbol, strength))

        relative_strengths.sort(key=lambda x: x[1], reverse=True)
        picks = [pair[0] for pair in relative_strengths[: self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest