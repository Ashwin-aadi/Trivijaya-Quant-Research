from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are generally more stable and less prone to extreme movements. "
        "By tilting towards low-volatility stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        volatilities: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in history.columns or "symbol" in history.columns and len(history[history["symbol"] == symbol].select("adj_close").to_numpy().flatten()) < self._window:
                continue
            values = [float(v) for v in history[symbol].select("adj_close").drop_nulls().to_list()]
            if len(values) < self._window:
                continue
            volatility = (pl.Series(values).std() * 252**0.5).item()
            volatilities[symbol] = volatility

        sorted_symbols = [k for k, v in sorted(volatilities.items(), key=lambda item: item[1])]
        
        if not sorted_symbols:
            return Signal(information_available_at=stamp, weights={})

        top_n = 5
        weight = 1.0 / top_n
        selected_symbols = sorted_symbols[:top_n]
        weights = {s: weight for s in selected_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max().item()
    assert isinstance(newest, date)
    return newest