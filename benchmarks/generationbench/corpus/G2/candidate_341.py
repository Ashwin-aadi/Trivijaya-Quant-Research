from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class BreakoutContinuation(Strategy):
    rationale = (
        "Breakouts often lead to continuation of price movement. By identifying symbols that "
        "have recently broken out and then continued in the same direction for a certain period,"
        " we can capture this momentum."
    )

    def __init__(self, window: int = 20, continuation_period: int = 5) -> None:
        self._window = window
        self._continuation_period = continuation_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._continuation_period)
        if history.height < self._window + self._continuation_period:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols: list[str] = []
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            values = [float(v) for v in history[symbol].to_list()]
            if len(values) < self._window + self._continuation_period:
                continue

            # Check for breakout condition
            last_close = values[-1]
            max_price = max(values[: self._window])
            min_price = min(values[: self._window])

            # Breakout condition: close at the top or bottom of recent range
            if (last_close > max_price and last_close - max_price >= 0.05 * (max_price - min_price)) \
                    or (last_close < min_price and min_price - last_close >= 0.05 * (max_price - min_price)):
                # Check for continuation in the same direction
                trend = pl.col("close").shift(-self._continuation_period).sort(descending=True).head(1) > \
                        pl.col("close").shift(-self._continuation_period - 1).sort(descending=True).head(1)
                if trend:
                    breakout_symbols.append(symbol)

        # Limit to top_n symbols
        breakout_symbols = breakout_symbols[: self._continuation_period]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest