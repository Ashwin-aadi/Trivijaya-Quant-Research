from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalTrade(Strategy):
    rationale = (
        "Stock markets often exhibit seasonality where certain months of the year are more "
        "favorable for returns than others. For instance, December can see strong performance "
        "due to holiday-related buying or the January effect, where markets experience positive "
        "returns following a significant downturn in late December."
    )

    def __init__(self, lookback_period: int = 5, forward_period: int = 10) -> None:
        self._lookback_period = lookback_period
        self._forward_period = forward_period

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._lookback_period + self._forward_period)

        if closes.height < self._lookback_period + self._forward_period:
            return Signal(information_available_at=stamp, weights={})

        # Filter symbols that have data for the entire period
        valid_symbols = [
            symbol
            for symbol in view.symbols
            if all(closes.get_column(symbol).to_list()[-self._lookback_period - self._forward_period :])
        ]

        if not valid_symbols:
            return Signal(information_available_at=stamp, weights={})

        # Calculate returns over the lookback and forward periods
        returns = []
        for symbol in valid_symbols:
            close_series = closes.get_column(symbol).to_list()
            lookback_close = float(close_series[-self._lookback_period - 1])
            forward_close = float(close_series[-1])
            return_ = (forward_close - lookback_close) / lookback_close
            returns.append((symbol, return_))

        # Rank symbols by their return and select the top ones
        ranked_returns = sorted(returns, key=lambda x: x[1], reverse=True)
        top_symbols = [s for s, _ in ranked_returns[: self._forward_period]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest