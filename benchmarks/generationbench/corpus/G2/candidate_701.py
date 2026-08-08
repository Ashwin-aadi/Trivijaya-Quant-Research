from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Continuance of a breakout can be exploited by identifying stocks that have recently "
        "broken out and then continue to rise. The economic reasoning is that breakout signals "
        "indicate strong demand or supply, which often leads to sustained price movements."
    )

    def __init__(self, window: int = 20, continuation_lookback: int = 10) -> None:
        self._window = window
        self._continuation_lookback = continuation_lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_lookback)

        if history.height < self._window + self._continuation_lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(values) < self._window + self._continuation_lookback:
                continue

            breakout_price = max(values[-self._window :])
            continuation_period = values[-self._window - 1 : -1]

            if all(value > breakout_price for value in continuation_period):
                breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:20]  # Select top 20 symbols
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest