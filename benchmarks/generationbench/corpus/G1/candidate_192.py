from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency of stocks within an index to "
        "outperform over short periods if they have performed well relative to their peers. "
        "This strategy buys top performers and sells underperformers."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_list = [symbol for symbol in view.symbols if symbol in closes.columns]
        if not symbol_list:
            return Signal(information_available_at=stamp, weights={})

        close_series = pl.Series([closes[symbol][-1] for symbol in symbol_list], symbol_list)
        mean_close = close_series.mean()
        above_mean = [symbol for symbol, close in zip(symbol_list, close_series.to_list()) if close > mean_close]
        
        if len(above_mean) < self._top_n:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = sorted(above_mean, key=lambda x: closes[x].tail(self._window).mean(), reverse=True)[:self._top_n]

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest