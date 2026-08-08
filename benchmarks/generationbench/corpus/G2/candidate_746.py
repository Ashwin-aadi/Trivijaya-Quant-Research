from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "Trend following strategies exploit long-term trends by buying assets that have been "
        "on an upward trend and selling those on a downward trend. However, the effectiveness "
        "of these strategies can be enhanced by scaling trades based on volatility to reduce "
        "risk during volatile periods."
    )

    def __init__(self, window: int = 30, vol_window: int = 20) -> None:
        self._window = window
        self._vol_window = vol_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + self._vol_window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        symbols = [s for s in view.symbols if s in closes]

        volatilities: dict[str, float] = {
            symbol: _calculate_volatility(closes[history.get_column(symbol).to_list()[-self._vol_window :]])
            for symbol in symbols
        }

        trends: dict[str, int] = {
            symbol: _trend_direction(closes[history.get_column(symbol).to_list()[1 : self._window + 1]]) for symbol in symbols
        }

        signal_weights: dict[str, float] = {}

        for symbol in symbols:
            if abs(trends[symbol]) == 1 and volatilities[symbol] > 0.5 * max(volatilities.values()):
                weight = 1.0 / len(symbols)
                signal_weights[symbol] = weight if trends[symbol] == 1 else -weight

        return Signal(
            information_available_at=stamp,
            weights={s: w for s, w in signal_weights.items() if w != 0}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(closes: list[float]) -> float:
    returns = [c / closes[i - 1] - 1.0 for i, c in enumerate(closes) if i > 0]
    std_dev = (sum(returns) ** 2 / len(returns)) ** 0.5
    return std_dev


def _trend_direction(closes: list[float]) -> int:
    changes = [c2 - c1 for c1, c2 in zip(closes[:-1], closes[1:])]

    if all(change > 0 for change in changes):
        return 1
    elif all(change < 0 for change in changes):
        return -1
    else:
        return 0