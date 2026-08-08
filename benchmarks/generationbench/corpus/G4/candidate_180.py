from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class Momentum6m(Strategy):
    rationale = (
        "This strategy exploits the cross-sectional momentum in Indian equities by "
        "investing in stocks with strong historical performance over the last 6 months. "
        "It leverages investor behavior and market expectations to capture excess returns."
    )

    def __init__(self, window: int = 180, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        symbols = set(closes.columns) & set(history["symbol"].to_list())

        momentum_scores: dict[str, float] = {}
        for symbol in symbols:
            if symbol not in history["symbol"].to_list():
                continue
            prices = history.filter(pl.col("symbol") == symbol)[
                ["session_date", "adj_close"]
            ]
            returns = (
                (prices.with_column(prices["adj_close"] / prices["adj_close"].shift(1) - 1.0).select(
                    pl.col("adj_close").tail(self._window).alias("returns")
                )).select(pl.col("returns").sum().alias(f"return_{symbol}"))
            ).to_series()
            momentum_scores[symbol] = returns

        sorted_symbols = [s for s, _ in sorted(momentum_scores.items(), key=lambda x: -x[1])]
        top_symbols = sorted_symbols[: self._top_n]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest