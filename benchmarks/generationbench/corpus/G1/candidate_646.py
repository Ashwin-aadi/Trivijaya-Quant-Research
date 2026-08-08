from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffectStrategy(Strategy):
    rationale = (
        "Certain stocks exhibit stronger performance at specific times of the year due to "
        "seasonal factors such as fiscal year-end activities or regulatory events. By "
        "identifying these seasonal effects, we can predict and capitalize on periods of higher "
        "volatility and returns."
    )

    def __init__(self, window: int = 30) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        seasonality_factors: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            avg_return = sum((v - values[0]) / values[0] for v in values[-30:]) / len(values[-30:])
            seasonality_factors[symbol] = avg_return

        top_symbols = sorted(seasonality_factors.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        weights = {symbol: 0.2 for symbol, _ in top_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest