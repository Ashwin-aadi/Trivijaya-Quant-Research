from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends while adjusting positions based on recent volatility. "
        "High volatilities may reduce exposure or reverse signals to limit risk, whereas low volatilities allow for more aggressive entry."
    )

    def __init__(self, window: int = 20, volatility_window: int = 10) -> None:
        self._window = window
        self._volatility_window = volatility_window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty() or history.height < self._window + 1:
            return Signal(information_available_at=stamp, weights={})

        closes = [float(v) for v in history["adj_close"].to_list()]
        volatilities: list[float] = []
        for i in range(len(closes) - self._volatility_window):
            returns = [(closes[i + j] / closes[i + j - 1]) - 1.0 for j in range(1, self._volatility_window + 1)]
            volatilities.append(max(returns))

        symbol_volatility_map: dict[str, float] = {}
        for i, symbol in enumerate(view.symbols):
            if symbol not in closes:
                continue
            volatility = max([closes[i + j] / closes[i + j - 1] - 1.0 for j in range(1, self._volatility_window + 1)])
            symbol_volatility_map[symbol] = volatility

        top_symbols = sorted(symbol_volatility_map.keys(), key=lambda x: symbol_volatility_map[x], reverse=False)[:5]

        weights: dict[str, float] = {}
        if top_symbols:
            weight = 1.0 / len(top_symbols)
            for symbol in top_symbols:
                if symbol not in closes:
                    continue
                weights[symbol] = weight

        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest