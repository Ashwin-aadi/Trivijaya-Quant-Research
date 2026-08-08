from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "The low-volatility tilt effect suggests that stocks with lower volatility tend to outperform "
        "high-volatility stocks over the long term. This persistence is attributed to pricing inefficiencies "
        "and structural biases in the market, offering potential for reduced portfolio volatility and enhanced returns."
    )

    def __init__(self, window: int = 20, top_n: int = 30) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        symbols = [s for s in view.symbols if s in history["symbol"].to_list()]
        if len(symbols) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol).sort("session_date")
            prices = [float(v) for v in df["close"].to_list()]
            returns = [
                (prices[i + 1] - prices[i]) / prices[i]
                for i in range(len(prices) - 1)
            ]
            volatilities[symbol] = pl.DataFrame(returns).select(
                ((pl.col("value") * pl.col("value")).mean()).alias("volatility")
            ).collect().item()

        sorted_symbols = [
            s[0]
            for s in sorted(volatilities.items(), key=lambda x: float(x[1]))
        ][: self._top_n]

        weights = {s: 1.0 / len(sorted_symbols) for s in sorted_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest