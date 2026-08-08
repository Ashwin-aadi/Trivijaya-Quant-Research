from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Companies that consistently outperform the broader market may have superior management "
        "or business models. This strategy aims to identify such companies by comparing their "
        "performance against the NIFTY 100 index."
    )

    def __init__(self, window: int = 252) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if view.closes().height < (len(view.symbols) + 1):
            return Signal(information_available_at=stamp, weights={})

        nifty_closes = view.closes(symbol="^NSEI", lookback=self._window)
        stock_closes = view.closes(lookback=self._window)

        if nifty_closes.height < self._window or stock_closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        nifty_returns = (
            (nifty_closes["adj_close"] / nifty_closes["adj_close"].shift(1) - 1.0)
            .to_list()
            [1:]
        )
        stock_returns = [
            float(v)
            for v in (
                (stock_closes[view.symbols[0]] / stock_closes[view.symbols[0]].shift(1) - 1.0)
                .drop_nulls()
                .to_list()
            )[: self._window]
        ]

        if len(stock_returns) < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_nifty_return = sum(nifty_returns) / self._window
        avg_stock_return = sum(stock_returns) / self._window

        if avg_stock_return > (avg_nifty_return + 0.1 * abs(avg_nifty_return)):
            return Signal(
                information_available_at=stamp,
                weights={view.symbols[0]: 1.0},
            )
        else:
            return Signal(information_available_at=stamp, weights={})


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest