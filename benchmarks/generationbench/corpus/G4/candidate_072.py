from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum4to12Weeks(Strategy):
    rationale = (
        "This strategy exploits cross-sectional momentum by identifying stocks that have performed well "
        "recently and are expected to continue outperforming. It leverages the persistence of short-term stock "
        "performance due to factors such as investor behavior, information delays, and market inefficiencies."
    )

    def __init__(self, window_min: int = 4, window_max: int = 12, top_n: int = 25) -> None:
        self._window_min = window_min
        self._window_max = window_max
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window_max)

        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate cumulative returns for each stock
        closes = view.closes(lookback=self._window_max)
        cum_returns: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue

            prices = [float(v) for v in history.filter(pl.col("session_date") <= stamp).select(pl.col(symbol)).to_numpy().flatten()]
            if len(prices) < self._window_min or all(price == prices[0] for price in prices):
                continue
            cum_return = (prices[-1] / prices[0]) - 1.0
            cum_returns[symbol] = cum_return

        # Rank stocks based on cumulative returns
        ranked_symbols = sorted(cum_returns.keys(), key=lambda s: cum_returns[s], reverse=True)
        top_n_symbols = ranked_symbols[: self._top_n]

        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest