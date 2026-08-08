from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Firms with higher relative strength compared to the broader market "
        "are likely outperforming and are expected to continue this trend due to "
        "positive sentiment or better fundamental conditions. This strategy aims to "
        "identify such firms by comparing their performance against the NIFTY 100 index."
    )

    def __init__(self, window: int = 50) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.latest_close()["^NSEI"]
        symbol_closes = {symbol: float(close) for symbol, close in closes.to_dict().items() if close != "NA"}
        
        # Calculate relative strength
        rel_strength = {
            symbol: (close / market_close - 1.0) * 100.0
            for symbol, close in symbol_closes.items()
        }
        
        # Sort by relative strength
        sorted_symbols = sorted(rel_strength.keys(), key=lambda x: rel_strength[x], reverse=True)
        
        top_n = min(self._window, len(sorted_symbols))
        picks = sorted_symbols[:top_n]
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in picks},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest