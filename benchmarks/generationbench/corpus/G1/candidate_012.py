from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed well "
        "in the recent past to continue outperforming. By investing in top performers, we aim "
        "to capture this positive momentum."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        momentum_scores = pl.DataFrame()
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            returns = [(values[i] / values[i - 1] - 1.0) for i in range(1, len(values))]
            momentum_score = sum(returns[-self._window:])
            momentum_scores = (
                momentum_scores.with_column(pl.Series([symbol], [momentum_score]))
                if not momentum_scores.is_empty()
                else pl.DataFrame({"symbol": [symbol], "score": [momentum_score]})
            )

        if momentum_scores.height < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = momentum_scores.sort("score", descending=True)["symbol"].to_list()[: self._top_n]
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