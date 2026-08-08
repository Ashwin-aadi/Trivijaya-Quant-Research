from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Reversion20d(Strategy):
    rationale = (
        "Reversion to the mean suggests that prices which have deviated significantly from their "
        "historical average should tend to revert. By identifying such deviations and trading "
        "against them, we can exploit this tendency for profit."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        avg_close = closes.mean().to_dict()["adj_close"]
        deviations = [
            (symbol, abs(float(close) - avg_close))
            for symbol, close in zip(view.symbols, closes["session_date", "adj_close"].to_dict()["adj_close"])
        ]

        # Identify top N symbols with the largest absolute deviations
        top_n_symbols = sorted(deviations, key=lambda x: x[1], reverse=True)[:5]
        picks = [symbol for symbol, _ in top_n_symbols]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest