from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilitySCTF(Strategy):
    rationale = (
        "This strategy identifies stocks with recent high volatility and trends them. High "
        "volatility often precedes price movements, suggesting that such stocks are more likely "
        "to break out or trend significantly."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window * 2 + 1:
            return Signal(information_available_at=stamp, weights={})

        recent_closes = closes.drop_nulls()
        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in recent_closes.columns:
                continue
            values = [float(v) for v in recent_closes[symbol].to_list()]
            returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            volatility = (sum([abs(r) for r in returns]) / len(returns)) ** 0.5
            volatilities[symbol] = volatility

        sorted_symbols = sorted(volatilities.items(), key=lambda x: x[1], reverse=True)
        picks = [symbol for symbol, _ in sorted_symbols[:self._top_n]]

        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest