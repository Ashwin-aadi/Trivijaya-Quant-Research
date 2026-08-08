from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks tend to outperform high-volatility stocks over the long term. "
        "This strategy aims to tilt towards low volatility by selecting the least volatile "
        "stocks from the market."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or len(history["symbol"].unique()) < 5:
            return Signal(information_available_at=stamp, weights={})

        volatilities = []
        for symbol in view.symbols:
            if symbol not in history.select("symbol").to_pandas().set_index("symbol"):
                continue
            closes = [float(v) for v in history.filter(pl.col("symbol") == symbol)["adj_close"].to_list()]
            if len(closes) < self._window:
                continue
            returns = [(closes[i + 1] - closes[i]) / closes[i] for i in range(len(closes) - 1)]
            volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
            volatilities.append((symbol, volatility))

        if not volatilities:
            return Signal(information_available_at=stamp, weights={})

        sorted_volatilities = sorted(volatilities, key=lambda x: x[1])
        top_symbols = [x[0] for x in sorted_volatilities[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest