from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Identifying stocks that outperform the broader market can provide excess returns. "
        "This strategy selects the top N performing stocks based on their cumulative return against the market index."
    )

    def __init__(self, window: int = 20, top_n: int = 5) -> None:
        self._window = window
        self._top_n = top_n

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        history = view.history(lookback=self._window)
        if history.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        market_index = view.symbols[0]
        index_closes = [float(v) for v in view.closes()[market_index].drop_nulls().to_list()]
        if len(index_closes) < self._window or not all(isinstance(x, (int, float)) for x in index_closes):
            return Signal(information_available_at=stamp, weights={})

        symbol_returns: dict[str, float] = {}
        for symbol in view.symbols[1:]:
            if symbol not in history.columns:
                continue
            symbol_closes = [float(v) for v in history[symbol].drop_nulls().to_list()]
            if len(symbol_closes) < self._window or not all(isinstance(x, (int, float)) for x in symbol_closes):
                continue

            cumulative_return = (symbol_closes[-1] / index_closes[0]) - 1.0
            symbol_returns[symbol] = cumulative_return

        sorted_symbols = sorted(symbol_returns.items(), key=lambda x: x[1], reverse=True)
        picks = [s for s, _ in sorted_symbols[: self._top_n]]

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