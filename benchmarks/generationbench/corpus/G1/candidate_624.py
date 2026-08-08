from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the idea that assets with positive past returns "
        "are likely to continue outperforming in the near future. This strategy identifies "
        "high-returning stocks and allocates capital accordingly."
    )

    def __init__(self, window: int = 20, lookback: int = 5) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._lookback)
        if history.height < self._lookback * 200:
            return Signal(information_available_at=stamp, weights={})

        closes = history.to_pandas().set_index("session_date")["adj_close"].unstack()
        returns = (closes / closes.shift(1) - 1).dropna(axis=1)
        
        mean_returns = returns.rolling(window=self._window).mean().iloc[-1]
        sorted_symbols = mean_returns.sort_values(ascending=False).index[:5]

        if not sorted_symbols.size:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest