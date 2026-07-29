from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingMean(Strategy):
    rationale = (
        "Price levels that revert to the mean over a trailing period indicate "
        "momentum decay and potential correction. This strategy aims to identify such levels."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_reversions: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            trailing_mean = sum(adj_closes[-self._window:]) / self._window
            current_price = adj_closes[-1]
            reversion_score = abs(current_price - trailing_mean)
            symbol_reversions[symbol] = reversion_score

        top_symbols = sorted(symbol_reversions.keys(), key=lambda x: symbol_reversions[x])[:5]

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