from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum suggests that stocks with strong relative performance "
        "over the past period are likely to continue outperforming. This strategy aims to "
        "capitalize on such persistent trends."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns
        returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(self._window) - 1.0).alias("r")
            )
            .sort("session_date", descending=True)
            .select(pl.col("r"))
            .to_series()
            .to_list()
        )

        # Identify top performing stocks
        symbols = history["symbol"].to_list()
        if len(returns) < self._window:
            return Signal(information_available_at=stamp, weights={})

        top_symbols: list[str] = []
        for i in range(self._window - 1, len(symbols)):
            symbol = symbols[i]
            rank = returns[i - (self._window - 1)].rank(method="ordinal", descending=True)
            if rank <= self._top_n:
                top_symbols.append(symbol)

        weights: dict[str, float] = {s: 1.0 / len(top_symbols) for s in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest