from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion suggests that stock prices which have deviated significantly from their "
        "long-term average will eventually return to it. In a short-horizon context, this means"
        "that stocks which are currently overvalued relative to their historical mean may revert"
        "to more typical valuations."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        signals: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            mean_price = sum(values) / len(values)
            recent_price = values[-1]
            reversion_factor = (mean_price - recent_price) / mean_price

            if reversion_factor > 0.2:
                signals[symbol] = 1.0 / len(signals)

        if not signals:
            return Signal(information_available_at=stamp, weights={})
        else:
            return Signal(
                information_available_at=stamp,
                weights={s: w for s, w in signals.items()},
            )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest