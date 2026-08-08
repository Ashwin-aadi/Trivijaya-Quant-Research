from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over long periods. "
        "This is because low-volatility stocks often benefit from stable earnings and lower risk, "
        "which can be rewarded with higher returns in the long run."
    )

    def __init__(self, window: int = 60) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        volumes = history["volume"].to_list()

        # Calculate daily returns
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = [sum([abs(ret) * volume for ret, volume in zip(returns[i-self._window:i], volumes[i-self._window:i])]) / sum(volumes[i-self._window:i]) for i in range(self._window, len(returns))]

        # Identify the lowest volatilities
        low_volatility_symbols = []
        for symbol in view.symbols:
            if symbol not in history.columns or "adj_close" not in history.columns:
                continue

            adj_closes = [float(c) for c in history[symbol]["adj_close"].drop_nulls().to_list()]
            vol = sum([abs(ret) * volume for ret, volume in zip(returns, volumes)]) / sum(volumes)
            if len(adj_closes) >= self._window and vol == min(volatility):
                low_volatility_symbols.append(symbol)

        # Ensure there are enough symbols to invest in
        if not low_volatility_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(low_volatility_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in low_volatility_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest