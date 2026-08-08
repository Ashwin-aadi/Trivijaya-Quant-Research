from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following exploits the tendency of assets with higher volatility "
        "to exhibit mean reversion. By scaling trends by their recent volatilities, we can identify "
        "assets that are currently in a strong uptrend relative to their historical volatility and "
        "allocate more capital to them."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbol_data = {}
        for symbol in view.symbols:
            if symbol not in history.columns:
                continue
            close_values = [float(v) for v in history[symbol].to_list()]
            if len(close_values) < self._window:
                continue

            # Calculate daily returns
            returns = [(close_values[i] - close_values[i-1]) / close_values[i-1] for i in range(1, len(close_values))]

            # Calculate the volatility (std dev of returns)
            volatility = pl.Series(returns).std()

            # Calculate the last price and its z-score relative to historical prices
            latest_close = float(history[symbol][-1])
            z_score = (latest_close - pl.Series(close_values[-self._window:]).mean()) / volatility

            symbol_data[symbol] = {"z_score": z_score, "volatility": volatility}

        # Identify symbols with high positive z-scores
        selected_symbols = [s for s in symbol_data if symbol_data[s]["z_score"] > 2 and symbol_data[s]["volatility"] > 0]

        if not selected_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(selected_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in selected_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest