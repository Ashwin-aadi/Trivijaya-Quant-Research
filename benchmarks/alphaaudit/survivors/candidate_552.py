from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for stocks that have performed "
        "well in the recent past to continue outperforming. This strategy selects the top "
        "performers over a lookback period."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        symbols = [str(s) for s in view.symbols]
        recent_closes = closes[symbols].to_numpy().T.tolist()
        returns = [(close[-1] - close[0]) / close[0] for close in recent_closes]

        top_symbols = sorted(zip(view.symbols, returns), key=lambda x: x[1], reverse=True)[: self._top_n]
        top_symbols = [symbol for symbol, _ in top_symbols]

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