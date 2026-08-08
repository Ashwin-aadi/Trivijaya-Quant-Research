from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "By tilting towards low-volatility stocks, the strategy aims to minimize overall portfolio risk. "
        "Historical volatility over 20 days provides a measure of stock price fluctuations, which is used "
        "to identify and invest in less risky assets."
    )

    def __init__(self, window: int = 20, top_n: int = 20) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility: dict[str, float] = {}
        for symbol in view.symbols:
            df_symbol = history.filter(pl.col("symbol") == symbol)
            if df_symbol.is_empty():
                continue
            close_prices = [float(v) for v in df_symbol["adj_close"].drop_nulls().to_list()]
            volatility = _calculate_volatility(close_prices, self._window)
            symbol_volatility[symbol] = volatility

        sorted_symbols = sorted(symbol_volatility.items(), key=lambda x: x[1])
        picks = [symbol for symbol, _ in sorted_symbols[: self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 0.05
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(prices: list[float], window: int) -> float:
    mean_price = sum(prices[-window:]) / window
    variance = sum((price - mean_price) ** 2 for price in prices[-window:]) / (window - 1)
    volatility = (variance ** 0.5) * (252 ** 0.5)  # Annualizing the standard deviation
    return volatility