from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks based on their relative strength against the broader market can "
        "help identify outperformers. Stocks that are consistently outperforming the market "
        "may continue to outperform in the future."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or not view.symbols:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.latest_close()["^NSEI"]
        symbol_closes = {symbol: float(close) for symbol, close in closes.to_dict().items()}
        returns = {symbol: (close / market_close - 1.0) for symbol, close in symbol_closes.items()}

        top_n_symbols = sorted(returns.keys(), key=lambda x: returns[x], reverse=True)[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest