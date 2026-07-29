from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in the past to continue outperforming. This strategy ranks assets based on recent "
        "returns and allocates capital to the top performers."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = {symbol: float(close) for symbol, close in view.latest_close().items()}
        adj_closes = [float(v) for v in history["adj_close"].to_list()]
        returns = [(latest_closes[symbol] / prev - 1.0) for symbol, prev in zip(view.symbols, adj_closes)]

        top_n_indices = sorted(range(len(returns)), key=lambda i: returns[i], reverse=True)[:5]
        picks = [view.symbols[i] for i in top_n_indices]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest