from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CompositeVolatilityBreakout(Strategy):
    rationale = (
        "High volatility stocks are often subject to mean reversion. By identifying stocks "
        "with high 20-day realized volatility and then applying a breakout strategy on their "
        "short-term price action, we can capture potential reversions to the mean."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2:
            return Signal(information_available_at=stamp, weights={})

        picks: list[str] = []
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            vol = _rolling_volatility(values, self._window)
            max_close = max(values[-self._window:])
            min_close = min(values[-self._window:])
            if max_close - min_close >= 3 * (vol / len(values)):
                picks.append(symbol)

        picks = picks[: self._top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _rolling_volatility(values: list[float], window: int) -> float:
    returns = [values[i] / values[i - 1] - 1.0 for i in range(1, len(values))]
    volatility = (sum([r**2 for r in returns[-window:]]) / window) ** 0.5
    return volatility