from __future__ import annotations

from datetime import date

import polars as pl
from src.backtest.strategy import MarketView, Signal, Strategy


class LowVolatilityTilting(Strategy):
    rationale = (
        "Low-volatility stocks are often considered less risky and may exhibit more stable returns. "
        "By tilting our portfolio towards these stocks, we aim to reduce overall portfolio risk."
    )

    def __init__(self, window: int = 20, top_n_percentage: float = 0.1) -> None:
        self._window = window
        self._top_n_percentage = top_n_percentage

    def generate(self, view: MarketView) -> Signal:
        stamp = _latest_visible(view)
        closes = view.closes(lookback=self._window)
        if closes.height < self._window:
            return Signal(information_available_at=stamp, weights={})

        symbols = []
        for symbol in view.symbols:
            values = [float(v) for v in closes[symbol].drop_nulls().to_list()]
            if len(values) >= self._window:
                volatility = pl.DataFrame({"returns": [values[i] / values[i - 1] - 1.0 for i in range(1, self._window)]}).select(pl.col("returns").std()).item()
                symbols.append((symbol, volatility))

        if not symbols:
            return Signal(information_available_at=stamp, weights={})

        sorted_symbols = sorted(symbols, key=lambda x: x[1])
        top_n_count = int(len(sorted_symbols) * self._top_n_percentage)
        top_n_symbols = [s[0] for s in sorted_symbols[:top_n_count]]
        if not top_n_symbols:
            return Signal(information_available_at=stamp, weights={})

        weight = 1.0 / len(top_n_symbols)
        return Signal(
            information_available_at=stamp,
            weights={symbol: weight for symbol in top_n_symbols},
        )


def _latest_visible(view: MarketView) -> date:
    visible = view.history()
    if visible.is_empty():
        return date(1900, 1, 1)
    newest = visible["session_date"].max()
    assert isinstance(newest, date)
    return newest