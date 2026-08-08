from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in the recent past to continue performing well. By weighting recent winners "
        "heavily, we aim to capture this effect."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in closes.columns]
        close_series = [closes[symbol].drop_nulls() for symbol in symbols]

        recent_closes = [float(c[-1]) for c in close_series]
        recent_returns = [
            (c[-1] - float(c[0])) / float(c[0])
            for c in zip(*close_series)
        ]

        top_symbols = sorted(zip(recent_closes, recent_returns, symbols), key=lambda x: (-x[1], -x[0]))[:5]

        weights = {symbol: 0.2 for _, _, symbol in top_symbols}
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        return Signal(
            information_available_at=stamp, weights=dict(top_symbols)
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest