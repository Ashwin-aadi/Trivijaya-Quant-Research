from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "This strategy identifies stocks with the highest momentum across the market "
        "and allocates capital towards them. Momentum is calculated based on the "
        "percentage change in price over a short period."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_scores: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            latest_close = float(view.latest_close()[symbol])
            price_changes = (
                (closes[symbol].to_list()[-1]
                 - closes[symbol].shift(self._window).to_list()[-1])
                / closes[symbol].shift(self._window).to_list()[-1]
            )
            if not isinstance(price_changes, float):
                continue
            symbol_scores[symbol] = price_changes

        top_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest