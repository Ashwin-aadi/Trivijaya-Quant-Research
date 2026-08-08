from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks that are outperforming the broader market based on "
        "recent price trends can provide a basis for long positions. This strategy "
        "focuses on relative strength to capture potential winners."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)
        if closes.height < self._window or closes.width < len(view.symbols) + 1:
            return Signal(information_available_at=stamp, weights={})

        # Calculate daily returns
        returns = (
            history.with_columns(
                (pl.col("adj_close") / pl.col("adj_close").shift(1) - 1.0).alias("r")
            )
            .sort("session_date", descending=False)
            .select(pl.exclude(["symbol", "session_date"]))
            .to_numpy()
        )

        # Calculate mean return for the market
        market_mean = returns[:, 0].mean()

        # Filter symbols with data available
        valid_symbols = [
            symbol for symbol in view.symbols if symbol in closes.columns
        ]

        # Compute relative strength
        rel_strength = {
            symbol: (returns[closes[symbol].to_list(), 1:] - market_mean).sum()
            for symbol in valid_symbols
        }

        top_symbols = sorted(rel_strength.items(), key=lambda x: x[1], reverse=True)[
            : self._top_n
        ]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest