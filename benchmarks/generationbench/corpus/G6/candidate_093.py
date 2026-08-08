from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilt(Strategy):
    rationale = (
        "This strategy focuses on selecting stocks with the lowest 20-day volatility to "
        "construct a balanced portfolio that aims for reduced risk and potentially higher "
        "returns through diversification."
    )

    def __init__(self, window: int = 20, top_n_percentage: float = 0.4) -> None:
        self._window = window
        self._top_n_percentage = top_n_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            returns = [float(v) for v in (closes[symbol].shift(-1) / closes[symbol] - 1.0).drop_nulls().to_list()]
            if len(returns) < self._window:
                continue
            vol = pl.Series(returns).std()
            volatilities[symbol] = float(vol)

        sorted_symbols = sorted(volatilities.items(), key=lambda x: x[1])
        top_n_count = int(len(sorted_symbols) * self._top_n_percentage)
        picks = [symbol for symbol, _ in sorted_symbols[:top_n_count]]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in picks}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest