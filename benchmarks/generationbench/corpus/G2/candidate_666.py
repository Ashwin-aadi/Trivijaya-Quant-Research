from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength to the market are expected to outperform in "
        "bullish markets. This strategy identifies such stocks by comparing their returns "
        "against a broad market index (NIFTY 100) over a lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        if view.history().height < self._window * (len(view.symbols) + 1):
            return Signal(information_available_at=stamp, weights={})

        market_returns = (
            view.closes(lookback=self._window).select(
                pl.col("NIFTY 100").alias("market_return")
            ).with_columns(
                (pl.col("market_return") / pl.col("market_return").shift(1) - 1.0).alias("r")
            ).sort("session_date", descending=False).tail(self._window)
        )["r"].to_list()

        symbol_returns = {}
        for symbol in view.symbols:
            closes = view.closes(lookback=self._window)[symbol]
            if closes.is_empty():
                continue
            returns = [
                float(r) for r in (closes / closes.shift(1) - 1.0).to_list()[:-1]
            ]
            symbol_returns[symbol] = sum(returns)

        top_symbols = sorted(
            symbol_returns.items(), key=lambda x: x[1], reverse=True
        )[:5]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s, _ in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest