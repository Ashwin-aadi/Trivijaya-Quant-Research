from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "This strategy selects stocks that have outperformed the broad market in recent "
        "trading periods. It assumes that strong relative performance is a sign of good underlying fundamentals."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        broad_market = closes.mean().item()
        strengths: dict[str, float] = {}

        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            performance = sum([v - broad_market for v in values]) / len(values)
            strengths[symbol] = performance

        top_symbols = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:5]
        weights = {symbol: 1.0 / len(top_symbols) for symbol, _ in top_symbols}
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, weight in weights.items()},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().cast(pl.Date).item()
    assert isinstance(newest, date)
    return newest