from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityScaledTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to capture trends by adjusting position sizes "
        "based on recent volatility. High volatility periods indicate a need for smaller positions "
        "to reduce risk, while low volatility suggests larger positions can be taken."
    )

    def __init__(self, window: int = 20, factor: float = 1.5) -> None:
        self._window = window
        self._factor = factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)

        if history.is_empty() or history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = [float(v) for v in history["adj_close"].to_list()]
        daily_returns = [(recent_closes[i] - recent_closes[i-1]) / recent_closes[i-1]
                         for i in range(1, len(recent_closes))]
        
        volatility = pl.Series(daily_returns).abs().mean()
        momentum = (recent_closes[-1] - recent_closes[0]) / sum(abs(x) for x in daily_returns)
        
        signal_strength = 1.0 if momentum > 0 else -1.0
        risk_adjusted_return = signal_strength * volatility * self._factor

        symbol_weights = {symbol: abs(risk_adjusted_return) / len(view.symbols) for symbol in view.symbols}
        return Signal(information_available_at=stamp, weights=symbol_weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest