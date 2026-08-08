from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class MeanReversion(Strategy):
    rationale = (
        "Short-horizon mean reversion exploits the tendency of stock prices to revert "
        "to their historical average. By identifying stocks that have deviated significantly "
        "from their mean price and betting on a return to normal levels, we can capture "
        "arbitrage opportunities."
    )

    def __init__(self, window: int = 5) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = {symbol: float(close) for symbol, close in zip(view.symbols, view.latest_close().values())}
        
        mean_prices = (
            history
            .group_by("symbol")
            .agg((pl.col("adj_close").mean()).alias("mean_price"))
        )
        
        deviations = (latest_closes[symbol] - mean_prices.filter(pl.col("symbol") == symbol)["mean_price"].item() for symbol in view.symbols)
        sorted_symbols = [s[0] for s in sorted(zip(view.symbols, deviations), key=lambda x: abs(x[1]), reverse=True)[:5]]

        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in sorted_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest