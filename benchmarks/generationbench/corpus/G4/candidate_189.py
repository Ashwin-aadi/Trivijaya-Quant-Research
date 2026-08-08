from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "This strategy exploits the empirical observation that low-volatility stocks tend to outperform high-volatility "
        "stocks over long periods. By constructing a portfolio weighted towards low-volatility stocks and maintaining "
        "adequate diversification, we aim to achieve better risk-adjusted returns."
    )

    def __init__(self, window: int = 60, num_stocks: int = 30) -> None:
        self._window = window
        self._num_stocks = num_stocks

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = view.symbols
        volatilities = self._calculate_volatility(symbols, history)

        sorted_symbols = [
            symbol for _, symbol in sorted(volatilities.items(), key=lambda item: item[1])
        ][: self._num_stocks]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(sorted_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in sorted_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(symbols: tuple[str, ...], history: pl.DataFrame) -> dict[str, float]:
    volatilities = {}
    for symbol in symbols:
        if symbol not in history.columns:
            continue
        prices = [float(v) for v in history[symbol].to_list()]
        returns = [(pl.col("close") / pl.col("close").shift(1) - 1.0).alias("r") for i in range(self._window)]
        volatility = (pl.DataFrame(returns)).select(pl.col("r").std().alias("volatility"))[0, 0]
        volatilities[symbol] = float(volatility)
    return volatilities