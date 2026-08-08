from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "Volatility-scaled trend following (V-SCTF) exploits the tendency for assets to continue "
        "moving in their recent direction, but scaled by their volatility. This strategy "
        "enters a long position if the asset's price moves up significantly relative to its "
        "volatility over a lookback period and exits on subsequent retracements."
    )

    def __init__(self, window: int = 20, threshold_multiplier: float = 1.5) -> None:
        self._window = window
        self._threshold_multiplier = threshold_multiplier

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_prices = {}
        for symbol in view.symbols:
            symbol_history = history.filter(pl.col("symbol") == symbol).select(
                "session_date", "adj_close"
            )
            close_series = [float(v) for v in symbol_history["adj_close"].to_list()]
            price_change = (close_series[-1] - close_series[0]) / close_series[0]
            volatility = _calculate_volatility(close_series)
            if volatility > 0:
                threshold = self._threshold_multiplier * volatility
                if price_change > threshold:
                    symbol_prices[symbol] = close_series

        signals: dict[str, float] = {}
        for symbol, prices in symbol_prices.items():
            last_close = float(prices[-1])
            previous_close = float(prices[-2])
            trend_signal = (last_close - previous_close) / previous_close
            if abs(trend_signal) < 0.1:
                continue  # Avoid very weak signals

            weight = 1.0 / len(symbol_prices)
            signals[symbol] = weight

        return Signal(information_available_at=stamp, weights=signals)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest


def _calculate_volatility(prices: list[float]) -> float:
    mean_price = sum(prices) / len(prices)
    variance = sum((price - mean_price) ** 2 for price in prices) / (len(prices) - 1)
    volatility = (variance ** 0.5) / mean_price
    return volatility