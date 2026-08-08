from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class SeasonalEffect(Strategy):
    rationale = (
        "Certain stocks in the Indian market exhibit stronger performance during specific times of the year. "
        "By identifying these seasonal patterns, we can construct a strategy to capture higher returns."
    )

    def __init__(self, window: int = 180) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes()
        symbols = [col for col in closes.columns if col not in ["session_date"]]
        
        seasonal_signals: dict[str, float] = {}
        
        for symbol in symbols:
            df = history.filter(pl.col("symbol") == symbol)
            close_values = [float(v) for v in df["adj_close"].drop_nulls().to_list()]
            
            if len(close_values) < 10: 
                continue
            
            latest_closes = close_values[-30:]  # Adjusted to a longer window
            seasonal_pattern = max(latest_closes)
            
            if seasonal_pattern == close_values[-1]:
                seasonal_signals[symbol] = 1.0

        filtered_symbols = [s for s in seasonal_signals.keys() if seasonal_signals[s] > 0]
        
        if not filtered_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(filtered_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in filtered_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest