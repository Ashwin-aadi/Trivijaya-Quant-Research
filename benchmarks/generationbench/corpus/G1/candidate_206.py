from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class VolatilityTrendFollowing(Strategy):
    rationale = (
        "Volatility-scaled trend following aims to identify and ride strong trends while "
        "adjusting position size based on the recent volatility. High volatility indicates "
        "greater uncertainty and risk, so we adjust our positions accordingly."
    )

    def __init__(self, window: int = 20, factor: float = 1.5) -> None:
        self._window = window
        self._factor = factor

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_volatility = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            daily_returns = [(v - values[i-1]) / values[i-1] for i, v in enumerate(values) if i > 0]
            volatility = pl.Series(daily_returns).std()
            symbol_volatility[symbol] = volatility

        top_symbols = sorted(symbol_volatility.keys(), key=lambda x: abs(symbol_volatility[x]), reverse=True)[:5]

        weight = self._factor / len(top_symbols)
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