from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class ContinuationBreakout(Strategy):
    rationale = (
        "After a breakout, continued strength in volume and price suggests that the initial "
        "break was not just a one-day fluke but part of a longer-term trend. This strategy aims "
        "to identify such continuations."
    )

    def __init__(self, window: int = 20, min_volume_increase: float = 1.5) -> None:
        self._window = window
        self._min_volume_increase = min_volume_increase

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        breakout_symbols = []
        for symbol in view.symbols:
            row = history.filter(pl.col("symbol") == symbol).sort("session_date")
            if row.height < 20:
                continue
            last_close = float(row.select("close").last().item())
            first_close = float(row.select("close").first().item())
            max_close = float(row.with_column((pl.col("close").max()).alias("max")).select("max").item())

            if row.filter(pl.col("close") == max_close).height > 1 and last_close == max_close:
                volume_ratio = float(row.filter(pl.col("symbol") == symbol)["volume"].last() / 
                                     row.filter(pl.col("symbol") == symbol)["volume"].first())
                if volume_ratio >= self._min_volume_increase:
                    breakout_symbols.append(symbol)

        breakout_symbols = breakout_symbols[:5]
        if not breakout_symbols:
            return Signal(information_available_at=stamp, weights={})
        weight = 1.0 / len(breakout_symbols)
        return Signal(
            information_available_at=stamp, 
            weights={s: weight for s in breakout_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible.select("session_date").max().item()
    assert isinstance(newest, date)
    return newest