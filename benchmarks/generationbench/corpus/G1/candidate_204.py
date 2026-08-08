from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class CrossSectionalMomentum(Strategy):
    rationale = (
        "Cross-sectional momentum exploits the tendency for top performers across the market "
        "to continue performing well in the near future. By investing in the best-performing "
        "stocks, we aim to capture these excess returns."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window * len(view.symbols):
            return Signal(information_available_at=stamp, weights={})

        closes = history["adj_close"].to_list()
        mean_closes = [sum(closes[i:i + len(view.symbols)]) / len(view.symbols) for i in range(0, len(closes), len(view.symbols))]
        
        top_symbols: list[str] = []
        for symbol in view.symbols:
            if history.select(pl.col(symbol).is_null()).row_count()[0] == 0:
                latest_close = float(history.filter(pl.col("session_date") == stamp).select(pl.col(symbol)).to_list()[0][0])
                mean_close = mean_closes[-1]
                momentum = (latest_close - mean_close) / mean_close
                if len(top_symbols) < self._top_n and momentum > 0:
                    top_symbols.append(symbol)

        top_symbols = [s for s in top_symbols if float(view.latest_close()[s]) != 0.0]
        
        if not top_symbols:
            return Signal(information_available_at=stamp, weights={})

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