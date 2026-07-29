from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks based on their relative strength against the broad market can "
        "identify outperformers and potentially capture excess returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        broad_market = closes[view.symbols[0]].to_list()  # Assuming first symbol is the broad market
        relative_strength: dict[str, float] = {}
        
        for symbol in view.symbols[1:]:
            if symbol not in closes.columns:
                continue
            
            stock_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(stock_closes) < self._window:
                continue

            avg_stock = sum(stock_closes[-self._window:]) / self._window
            avg_broad_market = sum(broad_market[-self._window:]) / self._window
            
            if avg_stock > avg_broad_market:
                relative_strength[symbol] = (avg_stock - avg_broad_market) / avg_broad_market

        picks = sorted(relative_strength.items(), key=lambda x: x[1], reverse=True)[:self._top_n]
        
        if not picks:
            return Signal(information_available_at=stamp, weights={})
        
        weight = 1.0 / len(picks)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in [p[0] for p in picks]}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest