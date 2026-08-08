from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have outperformed the market in the recent past based on their "
        "price strength relative to the NIFTY 100 index can help identify potentially undervalued "
        "or well-performing stocks."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_close = view.latest_close()["^NIFTY 50"]
        stock_closes = {symbol: float(close) for symbol, close in closes.to_dicts()}
        
        strengths: list[float] = []
        for symbol, close in stock_closes.items():
            if symbol == "^NIFTY 50":
                continue
            strength = (close - market_close) / market_close
            strengths.append(strength)
            
        top_strengths = sorted(strengths, reverse=True)[:self._top_n]
        picks: list[str] = [symbol for symbol, close in stock_closes.items() if (close - market_close) / market_close in top_strengths]

        picks = [p for p in picks if p != "^NIFTY 50"]
        
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