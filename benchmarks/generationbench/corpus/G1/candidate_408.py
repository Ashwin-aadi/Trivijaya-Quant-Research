from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the idea that stocks with strong recent performance "
        "are likely to continue performing well in the near future. This strategy selects top-performing "
        "stocks based on their returns over a short lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"]
        returns = (closes / closes.shift(1) - 1.0).drop_nulls().to_list()
        ranks = [float(r) for r in closes.rank(method="ordinal", descending=True).to_list()]

        top_symbols: list[str] = []
        for symbol, rank, return_ in zip(history["symbol"], ranks, returns):
            if not pl.is_nan(rank) and not pl.is_nan(return_):
                if rank <= 10:
                    top_symbols.append(symbol)

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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