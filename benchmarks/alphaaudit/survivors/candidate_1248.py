from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for recent winners to continue outperforming "
        "recent losers. By investing in top performers and shorting bottom performers, we aim to capture "
        "this excess return."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_rankings = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            return_ratio = (values[-1] - values[0]) / values[0]
            symbol_rankings[symbol] = return_ratio

        ranked_symbols = sorted(symbol_rankings.items(), key=lambda x: x[1], reverse=True)
        top_symbols, _ = zip(*ranked_symbols[: self._top_n])
        bottom_symbols, _ = zip(*ranked_symbols[-self._top_n :])

        weights = {s: 0.25 for s in top_symbols}
        short_weights = {s: -0.25 for s in bottom_symbols}

        weights.update(short_weights)
        if not weights:
            return Signal(information_available_at=stamp, weights={})
        
        return Signal(
            information_available_at=stamp,
            weights={k: v for k, v in weights.items() if v != 0},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest