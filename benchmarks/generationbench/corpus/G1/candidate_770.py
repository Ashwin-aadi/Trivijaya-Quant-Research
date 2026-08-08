from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "This strategy seeks to follow trends by identifying symbols with high relative "
        "volatility. High volatility can indicate strong momentum, and we aim to capitalize on "
        "such trends."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        volatility_scaled_trend = {}
        for symbol in view.symbols:
            if symbol not in history["symbol"].to_list():
                continue
            close_prices = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(close_prices) < self._window:
                continue

            log_returns = [(close_prices[i] / close_prices[i-1] - 1.0) for i in range(1, len(close_prices))]
            mean_return = sum(log_returns) / len(log_returns)
            std_deviation = (sum([(r - mean_return)**2 for r in log_returns]) / len(log_returns))**0.5
            volatility_scaled_trend[symbol] = mean_return / std_deviation

        sorted_symbols = [symbol[0] for symbol in sorted(volatility_scaled_trend.items(), key=lambda item: item[1], reverse=True)]
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n_symbols = sorted_symbols[:5]
        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in top_n_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest