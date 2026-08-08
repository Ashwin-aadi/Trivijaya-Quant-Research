from __future__ import annotations

from datetime import date
import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Selecting stocks that have outperformed the broad market index (NIFTY 100) over a "
        "recent period can provide an edge in equity markets. The relative strength strategy "
        "identifies such outperformers and allocates capital accordingly."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.is_empty():
            return Signal(information_available_at=stamp, weights={})

        nifty100_index_closes = [float(v) for v in history["NIFTY 100"].to_list()]
        if len(nifty100_index_closes) < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbol_closes = {symbol: float(v) for symbol, v in view.closes(lookback=self._window).iter_rows()}
        
        relative_strength_scores = [
            (symbol, (close - nifty100_index_closes[-1]) / nifty100_index_closes[-1])
            for symbol, close in symbol_closes.items()
        ]
        
        sorted_scores = sorted(relative_strength_scores, key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_scores[:5]]

        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest