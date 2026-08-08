from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength against a broad market index are expected to outperform "
        "over the medium term. This is based on the idea that stocks that have consistently performed well "
        "relative to their peers are more likely to continue performing positively."
    )

    def __init__(self, window: int = 30, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window or "NIFTY 100" not in [s.upper() for s in view.symbols]:
            return Signal(information_available_at=stamp, weights={})

        # Calculate the cumulative returns
        nifty_100_history = history.filter(pl.col("symbol").upper().eq("NIFTY 100"))
        other_stocks_history = history.filter(pl.col("symbol").is_not_in(["NIFTY 100"]))

        nifty_100_returns = (
            (nifty_100_history["adj_close"].shift(-1) / nifty_100_history["adj_close"]) - 1.0
        ).to_list()
        other_stocks_returns = []
        for symbol in view.symbols:
            if symbol.upper() != "NIFTY 100":
                stock_returns = (
                    (other_stocks_history.filter(pl.col("symbol").upper().eq(symbol))["adj_close"]
                     .shift(-1) / other_stocks_history.filter(pl.col("symbol").upper().eq(symbol))["adj_close"]) - 1.0
                ).to_list()
                if len(stock_returns) == self._window:
                    other_stocks_returns.append(stock_returns)

        # Compute the relative strength of each stock
        relative_strength = []
        for i in range(len(other_stocks_returns)):
            stock_returns = other_stocks_returns[i]
            stock_avg_return = sum(stock_returns) / self._window
            nifty_avg_return = sum(nifty_100_returns) / self._window
            relative_strength.append(stock_avg_return / nifty_avg_return)

        top_n_indices = sorted(range(len(relative_strength)), key=lambda i: relative_strength[i], reverse=True)[:self._top_n]
        weight = 1.0 / len(top_n_indices)
        return Signal(
            information_available_at=stamp,
            weights={view.symbols[i]: weight for i in top_n_indices}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest