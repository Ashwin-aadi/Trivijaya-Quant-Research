from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in the recent past to continue outperforming those that have underperformed. "
        "This strategy allocates capital to the top performers based on their recent returns."
    )

    def __init__(self, lookback_window: int = 20) -> None:
        self._lookback_window = lookback_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_window)

        if closes.height < self._lookback_window:
            return Signal(information_available_at=stamp, weights={})

        recent_returns = (closes[closes.columns[1:]] / closes[closes.columns[0]].shift(1) - 1.0).to_series()
        top_symbols = recent_returns.max_rows(self._lookback_window // 5)
        
        if len(top_symbols) == 0:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
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