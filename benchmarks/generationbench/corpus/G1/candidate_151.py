from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RangeCompression(Strategy):
    rationale = (
        "Range compression indicates a period of consolidation where volatility has decreased. "
        "During such periods, the market may be preparing for a breakout or significant move in either direction."
    )

    def __init__(self, window: int = 20, threshold: float = 0.1) -> None:
        self._window = window
        self._threshold = threshold

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window + 5)
        if history.height < self._window + 5:
            return Signal(information_available_at=stamp, weights={})

        symbols = [symbol for symbol in view.symbols if history.get_column("symbol").contains(symbol)]
        
        picks: list[str] = []
        for symbol in symbols:
            high = float(history.filter(pl.col("symbol") == symbol)["high"].max())
            low = float(history.filter(pl.col("symbol") == symbol)["low"].min())
            close_20d = float(history.filter(pl.col("symbol") == symbol)["close"][-1])
            
            range_compression = (high - low) / close_20d
            
            if range_compression <= self._threshold:
                picks.append(symbol)
        
        picks = picks[:5]
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