from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Mean reversion is an economic theory that asset prices and earnings tend to move back "
        "towards the long-term mean or average. In the context of stocks, if a stock's price has "
        "dropped significantly from its recent high, it may revert to its historical mean over "
        "time. This strategy seeks to exploit such deviations."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        recent_close = view.latest_close()
        mean_prices = {}
        for symbol in view.symbols:
            if symbol not in recent_close.keys() or symbol not in history.columns:
                continue
            prices = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(prices) < self._window:
                continue
            mean_price = sum(prices) / len(prices)
            mean_prices[symbol] = mean_price

        candidates: list[str] = []
        for symbol, price in mean_prices.items():
            latest_close = float(recent_close[symbol])
            if (latest_close - price) / price < -0.1:
                candidates.append(symbol)

        if not candidates:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(candidates)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in candidates}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest