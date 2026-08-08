from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "Breakouts from support or resistance levels often lead to continuation of the breakout "
        "direction. By identifying such breakouts and holding them for a short period, we can benefit "
        "from the momentum."
    )

    def __init__(self, window: int = 20, lookback: int = 10) -> None:
        self._window = window
        self._lookback = lookback

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._lookback)
        if history.height < self._window + self._lookback:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            symbol_history = history[symbol]
            recent_highs = symbol_history["high"].tail(self._window)
            recent_lows = symbol_history["low"].tail(self._window)

            # Find breakouts from the recent range
            breakout_conditions = (
                (symbol_history["close"] >= recent_highs.max())
                | (symbol_history["close"] <= recent_lows.min())
            )
            if any(breakout_conditions.to_list()[-self._lookback:]):
                continue

            # Check for continuation pattern in the lookback period
            continuation_conditions = (
                symbol_history["close"].sort(descending=True).tail(2)[0]
                > symbol_history["adj_close"].max()
            )
            if any(continuation_conditions.to_list()[-self._lookback:]):
                breakout_symbols.append(symbol)

        # Filter symbols based on the highest close in the continuation period
        breakout_symbols = sorted(breakout_symbols, key=lambda s: float(history[s]["close"].tail(self._lookback)[0]), reverse=True)[:5]
        
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