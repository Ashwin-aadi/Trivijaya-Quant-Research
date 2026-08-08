from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks with higher relative strength tend to outperform the broader market over "
        "longer time horizons. This is based on the idea that stocks in strong sectors or "
        "industries can maintain momentum even as overall market sentiment shifts."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) < 3:
            return Signal(information_available_at=stamp, weights={})

        # Compute the ratio of each stock's close to the NIFTY 100 index close
        nifty_100_close = view.closes(lookback=self._window)["^NSEI"].to_list()[-self._window:]
        ratios = [float(c / n) for c, n in zip(closes["^NSEI"], closes["^NSEI"])]
        
        # Filter out NIFTY 100 itself and symbols not present in the lookback period
        filtered_ratios = {s: r for s, r in zip(view.symbols, ratios) if s != "^NSEI" and len(ratios) == self._window}
        
        # Sort by highest ratio (strongest relative performance)
        sorted_symbols = sorted(filtered_ratios.keys(), key=lambda x: filtered_ratios[x], reverse=True)[:5]
        
        weights = {s: 1.0 / len(sorted_symbols) for s in sorted_symbols}
        return Signal(information_available_at=stamp, weights=weights)


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest