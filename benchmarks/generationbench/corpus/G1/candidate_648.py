from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum strategy exploits the tendency for stocks that have "
        "performed well relative to the market in recent periods to continue outperforming. "
        "This is based on the assumption that past performance is indicative of future returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_values = {symbol: [] for symbol in view.symbols}
        for _, row in closes.iter_rows():
            for symbol, value in row.items():
                if pl.col(symbol).is_nan().any():
                    continue
                symbol_values[symbol].append(float(value))

        momentum_scores = {}
        for symbol, values in symbol_values.items():
            if len(values) < self._window:
                continue
            recent_closes = values[-self._window:]
            mean_recent_close = sum(recent_closes) / self._window
            momentum_score = (recent_closes[-1] - mean_recent_close) / mean_recent_close
            momentum_scores[symbol] = momentum_score

        top_symbols = sorted(momentum_scores, key=momentum_scores.get, reverse=True)[: self._top_n]
        if not top_symbols:
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