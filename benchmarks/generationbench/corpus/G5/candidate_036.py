from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have "
        "outperformed in recent periods to continue outperforming. This strategy "
        "identifies top performers and allocates capital accordingly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        def calculate_momentum(symbol: str) -> float:
            symbol_history = history.filter(pl.col("symbol") == symbol)
            if symbol_history.height < self._window:
                return 0.0
            latest_close = view.latest_close()[symbol]
            returns = [
                (closes[symbol].to_list()[i] - symbol_history["adj_close"].to_list()[i])
                / symbol_history["adj_close"].to_list()[i]
                for i in range(len(symbol_history))
            ]
            if all(r == 0.0 for r in returns):
                return 0.0
            momentum = max(returns)
            return float(momentum * latest_close)

        momentum_scores = [
            calculate_momentum(symbol)
            for symbol in view.symbols
        ]

        top_symbols = [symbol for _, symbol in sorted(
            zip(momentum_scores, view.symbols), reverse=True)[:self._top_n]
        ]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest