from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Stocks that have outperformed their peers in the broad market are more likely to continue "
        "outperforming due to positive momentum. This strategy aims to capture such stocks by "
        "selecting those with the highest relative strength over a lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        closes = view.closes(lookback=self._window)

        # Calculate relative strength for each stock
        rel_strength: dict[str, float] = {}
        for symbol in view.symbols:
            if symbol not in closes.columns:
                continue
            history_for_symbol = history[history["symbol"] == symbol]
            symbol_closes = [float(v) for v in history_for_symbol["adj_close"].to_list()]
            avg_close = sum(symbol_closes) / len(symbol_closes)
            # Calculate relative strength as the average return over the window
            avg_return = sum((close / avg_close - 1.0 for close in symbol_closes[-self._window:])) / self._window
            rel_strength[symbol] = avg_return

        # Sort symbols by their relative strength
        sorted_symbols = sorted(rel_strength.items(), key=lambda x: x[1], reverse=True)
        
        top_n_symbols = [symbol for symbol, _ in sorted_symbols[:5]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={s: weight for s in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest