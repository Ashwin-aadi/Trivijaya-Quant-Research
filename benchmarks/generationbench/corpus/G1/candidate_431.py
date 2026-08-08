from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeStrategy(Strategy):
    rationale = (
        "Combining two weakly related characteristics can sometimes generate more robust "
        "signals than relying on a single factor. This strategy looks at both 20-day momentum "
        "and recent volatility to determine potential breakout opportunities."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            if values[-1] >= max(values):
                breakout_symbols.append(symbol)

        volatility_symbols: list[str] = []
        history = view.history(lookback=self._window)
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            daily_returns = (history[symbol].slice(-2, 1) / history[symbol].slice(-3, 1) - 1.0).to_list()
            if len(daily_returns) < 2:
                continue
            volatility = abs(sum([r for r in daily_returns if not pl.col("r").is_nan()]))
            if volatility > 0.1:
                volatility_symbols.append(symbol)

        common_symbols = set(breakout_symbols).intersection(set(volatility_symbols))
        if not common_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(common_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in common_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest