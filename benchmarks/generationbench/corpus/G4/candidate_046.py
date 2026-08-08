from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalityEffect(Strategy):
    rationale = (
        "The 'January Effect' in the Indian market suggests that stocks often outperform "
        "in January due to liquidity constraints and tax considerations. This strategy aims "
        "to capitalize on this seasonal trend by going long on a diversified portfolio of "
        "equities from early January until mid-February."
    )

    def __init__(self, start_date: date = date(2020, 1, 1), end_date: date = date(2024, 2, 15)) -> None:
        self._start_date = start_date
        self._end_date = end_date

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if stamp < self._start_date or stamp > self._end_date:
            return Signal(information_available_at=stamp, weights={})

        history = view.history(lookback=365)
        symbols = [s for s in view.symbols if s in history.columns]

        if len(symbols) < 30:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=365).select(symbols)

        # Compute historical returns
        returns = (closes["2024-01"] / closes[symbols[0]].shift(365) - 1.0).to_series()
        ranked_stocks = returns.argsort().rank(method="dense", descending=True).to_list()

        top_n_stocks = [symbols[i] for i in ranked_stocks[:30]]
        weight = 1.0 / len(top_n_stocks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_n_stocks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest