from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ReversionToTrailingAverage(Strategy):
    rationale = (
        "Price reversion strategies exploit the tendency of financial markets to revert "
        "to their mean over time. This strategy identifies symbols whose current price is "
        "far from their trailing average and bets on a return towards this average."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        trailing_average: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            avg_value = sum(values[-self._window:]) / self._window
            trailing_average[symbol] = avg_value

        reversion_signals: list[str] = []
        for symbol, latest_close in view.latest_close().items():
            if symbol not in trailing_average:
                continue
            difference = (latest_close - trailing_average[symbol]) / trailing_average[symbol]
            if abs(difference) > 0.1:
                reversion_signals.append(symbol)

        weight = 1.0 / len(reversion_signals)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in reversion_signals}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest