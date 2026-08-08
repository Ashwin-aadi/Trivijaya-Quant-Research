from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Identifying stocks with the highest relative strength over a lookback period can "
        "help in capturing cross-sectional momentum effects. This strategy selects top-performing "
        "stocks based on their performance relative to the market index."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_index_close = max(closes["^NSEI"].to_list())
        stock_closes = {symbol: float(close) for symbol, close in zip(view.symbols, closes.drop_nulls().columns[1:]) if close}

        momentum_scores = [(symbol, (close / market_index_close - 1.0)) for symbol, close in stock_closes.items()]
        sorted_momentum_scores = sorted(momentum_scores, key=lambda x: x[1], reverse=True)

        picks = [score[0] for score in sorted_momentum_scores[: self._top_n]]
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest