from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class RelativeStrength(Strategy):
    rationale = (
        "Investing in assets that have outperformed the market provides a higher probability of "
        "positive returns. This strategy selects stocks with the highest relative strength over a "
        "lookback period."
    )

    def __init__(self, window: int = 20) -> None:
        self._window = window

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window or len(closes.columns) == 1:
            return Signal(information_available_at=stamp, weights={})

        latest_closes = view.latest_close()
        symbol_list = list(latest_closes.keys())

        # Calculate relative strength
        rel_strengths = []
        for symbol in symbol_list:
            if symbol not in closes.columns:
                continue

            # Get the adjusted close prices over the lookback period
            adj_closes = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(adj_closes) < self._window:
                continue

            # Calculate the relative strength as the percentage change from the start of the window to the end
            rel_strength = (adj_closes[-1] - adj_closes[0]) / adj_closes[0]
            rel_strengths.append((symbol, rel_strength))

        # Sort by relative strength in descending order
        sorted_strengths = sorted(rel_strengths, key=lambda x: x[1], reverse=True)

        if not sorted_strengths:
            return Signal(information_available_at=stamp, weights={})

        top_symbols = [symbol for symbol, _ in sorted_strengths[:5]]
        weight = 1.0 / len(top_symbols)
        return Signal(
            information_available_at=stamp, weights={s: weight for s in top_symbols}
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest