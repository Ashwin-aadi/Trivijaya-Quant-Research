from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Companies with higher relative strength outperform the broader market over time. "
        "This is often attributed to better fundamentals or superior management that allows "
        "them to generate consistent returns and attract more investment."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        returns = (
            (closes["adj_close"] / closes["adj_close"].shift(self._window) - 1.0).to_dict()
            | {"session_date": closes["session_date"]}
        )
        sorted_returns = pl.DataFrame(returns).sort("session_date", descending=True)

        symbols = [s for s in view.symbols if s in sorted_returns.columns]
        top_symbols = symbols[:5]  # Select the top 5 performers
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest